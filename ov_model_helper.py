# -*- coding: UTF-8 -*-
import gc
import numpy as np
from pathlib import Path
import os
import json
from typing import Optional, Tuple, Callable, Any, Union
import string
import random
import re
import types
import shutil
import torch
import torch.nn.functional as F
import yaml      

from transformers import AutoConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

import transformers
from packaging import version
transformers_ver = version.parse(transformers.__version__)

import openvino as ov
from openvino import save_model, convert_model
try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13
import nncf

RTOL_STRICT = 1e-3
ATOL_STRICT = 1e-3
equal_nan=True

def diff_mask_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = RTOL_STRICT,
    atol: float = ATOL_STRICT,
    equal_nan: bool = equal_nan,
    max_print: int = 10,
) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    """
    比较 a 与 b 的近似相等性（逐元素），并按“最后一维为列，其它维为行”的语义统计行相等情况。

    返回：
      - mask_neq: 与 a/b 同形的布尔张量，True 表示该元素“不近似相等”
      - num_neq: 不近似相等的元素总数（int）
      - num_equal_rows: 完全近似相等的“行”数（int），行的定义：除最后一维外的切片
      - row_equal_mask: shape 为 (∏a.shape[:-1],)，每个元素对应一行是否完全近似相等（bool）

    打印：
      - 前 max_print 个不近似相等的元素位置与数值差异
    """
    # ---- 基本校验 ----
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致：{a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"device 不一致：{a.device} vs {b.device}")

    # ---- 逐元素近似比较 ----
    close_mask = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    mask_neq = ~close_mask
    num_neq = int(mask_neq.sum().item())

    # ---- 打印前 max_print 个不近似相等的元素 ----
    if num_neq > 0 and max_print > 0:
        idxs = torch.nonzero(mask_neq, as_tuple=False)  # 每行是一个坐标
        k = min(num_neq, max_print)
        print(f"[diff] 不近似相等元素个数 = {num_neq}（展示前 {k} 个）")

        a_cpu = a.detach().cpu()
        b_cpu = b.detach().cpu()
        for i in range(k):
            idx_tuple = tuple(int(x) for x in idxs[i].tolist())
            va = a_cpu[idx_tuple].item()
            vb = b_cpu[idx_tuple].item()
            print(f"  #{i+1} 位置 {idx_tuple}: a={va}, b={vb}, |Δ|={abs(va - vb)}")

    # ---- 行定义：最后一维是列，其它全部维度展平为行 ----
    if a.ndim == 0:
        # 标量：没有“行/列”的概念，按 1 行 1 列处理
        row_equal_mask = close_mask.view(1)
        num_equal_rows = int(row_equal_mask.sum().item())
        print(f"[row] 标量视作 1 行：完全近似相等的行数={num_equal_rows}")
        return mask_neq, num_neq, num_equal_rows, row_equal_mask

    n_cols = a.shape[-1]
    n_rows = a.numel() // n_cols

    # 将前面所有维度展平成行，最后一维保留为列
    close_2d = close_mask.reshape(n_rows, n_cols)
    row_equal_mask = close_2d.all(dim=1)             # 每一行所有列都近似相等才算这一行相等
    num_equal_rows = int(row_equal_mask.sum().item())

    print(f"[row] 以最后一维为列：总行数={n_rows}，每行列数={n_cols}，完全近似相等的行数={num_equal_rows}")

    return mask_neq, num_neq#, num_equal_rows, row_equal_mask


# Below is function for OpenVINO patch stateful into models
def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0

def model_has_input_output_name(ov_model: ov.Model, name: str):
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
    input_batch_name: str,
):
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input(input_batch_name).get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

def build_state_initializer(ov_model: ov.Model, batch_dim: int, input_batch_name: str):
    input_ids = ov_model.input(input_batch_name)
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()

def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
    input_batch_name: str = "input_ids",
):
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim, input_batch_name)

def patch_stateful(ov_model, input_batch_name):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim, input_batch_name)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
        input_batch_name
    )
    
def cleanup_torchscript_cache():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
    gc.collect()

def patch_model_stateful(ov_model, input_names, output_names):
    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})
    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})
    patch_stateful(ov_model, input_names[0])
    return ov_model


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx

def prepare_padding_mask(attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask

def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask

def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask

def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or (kv_length == query_length or is_torch_xpu_available))
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (padding_mask is None or (padding_mask.all() if not is_torch_xpu_available or query_length == 1 else padding_mask[:, :query_length].all()))
    ):
        return True

    return False

def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset

    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask

# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
# Specifically for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
def eager_mask_without_vmap(*args, **kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    # we use torch.finfo(torch.float16).min instead torch.finfo(dtype).min to avoid an overflow but not
    # sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=mask.device, dtype=dtype),
        torch.tensor(torch.finfo(torch.float16).min, device=mask.device, dtype=dtype),
    )
    return mask

# for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
# Although I'm not sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

# for decoder models, we use eager mask without vmap for sdpa as well
# to avoid a nan output issue in OpenVINO that only happens in case of:
# non-stateful models on cpu and stateful models on npu
ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)


def patched_dynamic_layer_update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.keys is None:
        self.keys = key_states
        self.values = value_states
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True
    else:
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values

DynamicLayer.update = patched_dynamic_layer_update

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model_wrapper = model

    def forward(self, example_inputs):
        with torch.no_grad():
            return self.model_wrapper(**example_inputs)
    
    def convert_model(self, xml_Path, example_inputs, compress_weights=False):
        with torch.no_grad():
            ov_model = ov.convert_model(self, example_input=example_inputs)
            if compress_weights:
                ov_model = nncf.compress_weights(ov_model)
            ov.save_model(ov_model, xml_Path, compress_to_fp16=False)
            print(f"### save ov model @ {xml_Path}")
        # return self.forward(**example_inputs)
    
    def convert_onnx(self, onnx_Path, example_inputs, input_names, dynamic_axes):
        trace_model = torch.jit.trace(self, example_inputs)
        torch.onnx.export(trace_model, (), onnx_Path, input_names=input_names, dynamic_axes=dynamic_axes)

#simple wrapper for OpenVINO Model Convert
class EfficientMMOENetWrapper(ModelWrapper) :
    def __init__(self, model):
        super().__init__(model)
        
    def forward(self, anno, img):
        with torch.no_grad():
            preds = self.model_wrapper(anno, img)
            complete_score = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj0'].reshape((1, -1))))
            audits = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj1'].reshape((1, -1))))
            category_of_dishes = preds['obj2'].reshape((1, -1)).argmax(axis=-1)
            main_obvious_score = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj2'].reshape((1, -1))))
            # print(f"audits={audits}, map_score={map_score}, complete_score={complete_score}, category_of_dishes={category_of_dishes}, main_obvious_score={main_obvious_score}")
            return complete_score, audits, category_of_dishes, main_obvious_score

    def convert_model(self, inputs, xml_Path, compress_weights=False):
        example_inputs = {"x" : inputs}
        return super().convert_model(example_inputs, xml_Path, compress_weights)

    def convert_onnx(self, onnx_Path, inputs):
        for k,v in inputs.items() :
            print(f"{k}={v.shape}")
        input_names = [k for k in inputs.keys()]
        dynamic_axes = {'anno': { 1: 'length'},
                        'img': {  2: 'width', 3: 'height'},} 
        trace_model = torch.jit.trace(self, (inputs['anno'], inputs['img']))
        torch.onnx.export(trace_model, (inputs['anno'], inputs['img']), onnx_Path, 
                          input_names=input_names, dynamic_axes=dynamic_axes)

class ClipSegWrapper(ModelWrapper) :
    def __init__(self, model):
        super().__init__(model)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        with torch.no_grad():
            outputs = self.model_wrapper(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_dict=False)
            return outputs[0]
            cla = outputs.logits.argmax(axis=0)
            return cla
            pos = torch.mean(torch.argwhere(cla == 1).to(torch.float))
            pianyi = (torch.sum(((pos - torch.tensor([176, 176])) ** 2)) ** 0.5) / (torch.sum(((torch.tensor([176, 176])) ** 2)) ** 0.5) * 2
            auditsV2 = 1 / (1 + torch.exp(-torch.tensor([pianyi, audits]) @ torch.tensor([-2.10123607,  4.12227572]) + 2.33651427))
            return torch.tensor(auditsV2)
        
    def convert_model(self, xml_Path, inputs, compress_weights=False):
        example_inputs = {k:v for k,v in inputs.items()}
        for k,v in example_inputs.items() :
            print(f"{k}={v.shape}")
        return super().convert_model(xml_Path, example_inputs, compress_weights)
    
    def convert_onnx(self, onnx_Path, inputs):
        for k,v in inputs.items() :
            print(f"{k}={v.shape}")
        input_names = [k for k in inputs.keys()]
        dynamic_axes = {'input_ids': { 1: 'length',},
                        'attention_mask': { 1: 'length',},
                        'pixel_values': { 2: 'width', 3: 'height'}} 
        trace_model = torch.jit.trace(self, (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']))
        torch.onnx.export(trace_model,  (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']), onnx_Path, 
                          input_names=input_names, dynamic_axes=dynamic_axes)

# Enc-Dec model for OpenVINO Model Convert
# Encoder is simple mode
# Decoder always has kv-cache system 
# So we need to stateful model convert    
class FireRedAsrAedConverterWrapper() :
    def __init__(self, model):
        class ModelEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, feats, lengths):
                with torch.no_grad():
                    enc_outputs, _, enc_mask = self.model.encoder(feats, lengths)
                return enc_outputs, enc_mask

        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, t_ys, encoder_outputs, src_mask, softmax_smoothing, eos_penalty,
                            is_finished, B, N, scores, caches):
                    with torch.no_grad():
                        topB_row_number_in_ys, t_ys, scores, caches = self.model.decoder.infer_decoder(t_ys, 
                                            encoder_outputs, src_mask, caches, scores,
                                            softmax_smoothing, eos_penalty, is_finished, B, N)
                        return topB_row_number_in_ys, t_ys, scores, caches

        self.enc_wrapper = ModelEncoderWrapper(model)
        self.enc_wrapper.eval()
        self.dec_wrapper = ModelDecoderWrapper(model)
        self.dec_wrapper.eval()

    def convert_ov_model(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty,
                   ov_encoder_path, ov_decoder_path, sos_id, eos_id, pad_id, INF, quantization_config = None):
        if not ov_encoder_path.exists() :
            example_inputs = {"feats":feats, "lengths":lengths}
            ov_model = convert_model(self.enc_wrapper, example_input=example_inputs)
            save_model(ov_model, ov_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {ov_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()

        enc_outputs, enc_mask = self.enc_wrapper(feats, lengths)

        if not ov_decoder_path.exists() :
            beam_size=3
            num = 2
            cache_size = 16
            
            B = beam_size
            N, Ti, H = enc_outputs.size()
            cache_shape = (B*N, num, 1280)

            encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
            src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
            t_ys = torch.ones(N*B, 1).fill_(sos_id).long()
            scores = torch.tensor([0.0] + [-INF]*(B-1)).float()
            scores = scores.repeat(N).view(N*B, 1)
            is_finished = torch.zeros_like(scores)
            N = torch.tensor(N).long()
            B = torch.tensor(B).long()

            caches = []

            input_names = ["t_ys", "encoder_outputs", "src_mask", "softmax_smoothing",
                           "eos_penalty", "is_finished", "B", "N", "scores"]
            output_names = ["topB_row_number_in_ys", "new_t_ys", "new_scores"]

            for i in range(cache_size):
                cache = torch.randn(cache_shape)
                caches.append(cache)
                input_names.extend([f"key_values.{i}"])
                output_names.extend([f"present.{i}"])

            example_input = {"t_ys":t_ys, "encoder_outputs": encoder_outputs, "src_mask": src_mask,
                             "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty,
                             "is_finished":is_finished,"B": B, "N": N, 
                             "scores": scores, "caches": caches}
                
            ov_model = ov.convert_model(self.dec_wrapper, example_input=example_input)
            
            patch_model_stateful(ov_model, input_names, output_names)
            # for input, input_name in zip(ov_model.inputs, input_names):
            #     input.get_tensor().set_names({input_name})

            # for output, output_name in zip(ov_model.outputs, output_names):
            #     output.get_tensor().set_names({output_name})

            # patch_stateful(ov_model)
            print("✅ ModelDecoder model successfully converted")

            if quantization_config is not None and "llm" in quantization_config:
                print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
                print("✅ Weights compression finished")
            else:
                ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
            
            ov.save_model(ov_model, ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {ov_decoder_path}")

GLMASR_Audio_Encoder_MODEL_NAME = "glm_asr_audio_encoder.xml"
GLMASR_Input_Encoder_MODEL_NAME = "glm_asr_input_encoder.xml"
GLMASR_Encoder_MODEL_NAME = "glm_asr_encoder.xml"
GLMASR_Decoder_MODEL_NAME = "glm_asr_decoder.xml"
GLMASR_OV_CONFIG_NAME = "ov_config.yaml"

def to_legacy_cache(past_key_values):
    if transformers_ver.major >= 5:
        past_key_values_list = []
        for keys, values, attr  in past_key_values:
            past_key_values_list.append((keys, values))
        return past_key_values_list
    else :
        return past_key_values.to_legacy_cache()

def from_legacy_cache(past_key_values):
    if transformers_ver.major >= 5:
        past_key_values = DynamicCache(ddp_cache_data=past_key_values)
    else :
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values

class GlmAsrForOVConvertWrapper(GenerationMixin):
    _is_stateful = True   # or False
    _keep_in_fp32_modules_strict = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, model, processor, ov_model_path):
        super().__init__()
        # model.config._attn_implementation = "eager"
        self.processor = processor 
        self.config = model.config
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.device = torch.device("cpu")
        self.ov_core = None
        
        class ModelEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, input_features, input_features_mask):
                with torch.no_grad():
                    audio_embeds = self.model.get_audio_features(input_features, input_features_mask)
                    # from transformers.modeling_outputs import BaseModelOutputWithPooling
                    # if isinstance(audio_embeds, BaseModelOutputWithPooling) or hasattr(audio_embeds, 'pooler_output'):
                    if hasattr(audio_embeds, 'pooler_output'):
                        audio_embeds = audio_embeds.pooler_output
                return audio_embeds

        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.language_model = model.language_model.eval()

            def forward(self, input_ids, audio_embeds, audio_token_mask, attention_mask, position_ids, cache_position, past_key_values):
                with torch.no_grad():
                    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.dtype)
                        )
                    if isinstance(past_key_values, list) or isinstance(past_key_values, tuple):
                        past_key_values = from_legacy_cache(past_key_values)

                    result = self.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        labels=None,
                        use_cache=True,
                        cache_position=cache_position,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    past_key_values = to_legacy_cache(result.past_key_values)
                    return result.logits, past_key_values

        self.enc_wrapper = ModelEncoderWrapper(model)
        self.enc_wrapper.eval()

        self.dec_wrapper = ModelDecoderWrapper(model)
        self.dec_wrapper.eval()

        self.using_ov = False
        self.ov_core = None
        self.cache_size = 1000
        self.enc_type = 'bf16'
        self.dec_type = 'bf16'
        self.init_model_path(Path(ov_model_path))
        self.load_ov_model()

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path  / GLMASR_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path  / GLMASR_Decoder_MODEL_NAME
        self.ov_config_path = ov_path  / GLMASR_OV_CONFIG_NAME
        if not self.ov_encoder_path.exists() or not self.ov_config_path.exists():
            self.converted_to_ov = True
        
    def load_ov_model(self):
        try:
            if self.config is None  :
                self.config = AutoConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            if self.generation_config is None  :
                self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
        except Exception as e:
            print(f"### {e}")

        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]
            
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
           
            device = "CPU"
            ov_config = {}
            ov_config['NUM_STREAMS'] = 1
            ov_config['PERF_COUNT'] = 'NO'
            ov_config['INFERENCE_PRECISION_HINT'] = self.enc_type
            ov_config['PERFORMANCE_HINT'] = 'LATENCY'

            model = self.ov_core.read_model(self.ov_encoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.enc_request = compiled_model.create_infer_request()

            ov_config['INFERENCE_PRECISION_HINT'] = self.dec_type
            model = self.ov_core.read_model(self.ov_decoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.dec_request = compiled_model.create_infer_request()

            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} or {self.ov_config_path} failed, {e}")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs
   
    def __call__(self, *args, **kwargs):
        return self.forward(**kwargs)

    def forward(self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        if self.using_ov :
            return self.forward_both(input_ids=input_ids,
                                  input_features=input_features,
                                  input_features_mask=input_features_mask,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  cache_position=cache_position,
                                  logits_to_keep=logits_to_keep,
                                  **kwargs)
        else :
            return self.convert_to_ov(input_ids=input_ids,
                                  input_features=input_features,
                                  input_features_mask=input_features_mask,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  cache_position=cache_position,
                                  logits_to_keep=logits_to_keep,
                                  **kwargs)

    def forward_ov(self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        if input_features is not None:
            example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
            self.enc_request.start_async(example_inputs, share_inputs=True)
            self.dec_request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            self.enc_request.wait()
            audio_embeds = self.enc_request.get_output_tensor(0).data
        else :
            audio_embeds = torch.zeros((input_ids.shape[0], 1))
            audio_token_mask = torch.tensor([False]).reshape((input_ids.shape[0],1,1))


        example_inputs = {"input_ids":input_ids,
                          "audio_embeds":audio_embeds,
                          "audio_token_mask":audio_token_mask,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,
                          "beam_idx": self.next_beam_idx}
        self.dec_request.start_async(example_inputs, share_inputs=True)
        self.dec_request.wait()

        logits = torch.from_numpy(self.dec_request.get_tensor("logits").data)
        past_key_values = ((),)
        self._past_length += input_ids.shape[1]
        out = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        return out

    def forward_both(self, 
                      input_ids: torch.LongTensor | None = None,
                      input_features: torch.FloatTensor | None = None,
                      input_features_mask: torch.Tensor | None = None,
                      attention_mask: torch.Tensor | None = None,
                      position_ids: torch.LongTensor | None = None,
                      past_key_values = None,
                      inputs_embeds: torch.FloatTensor | None = None,
                      labels: torch.LongTensor | None = None,
                      use_cache: bool | None = None,
                      cache_position: torch.LongTensor | None = None,
                      logits_to_keep: int | torch.Tensor = 0,
                      **kwargs):
        if input_features is not None:
            example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
            audio_embeds1 = self.enc_wrapper(**example_inputs)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)

            self.enc_request.start_async(example_inputs, share_inputs=True)
            self.enc_request.wait()
            audio_embeds = torch.from_numpy(self.enc_request.get_output_tensor(0).data)

            self.dec_request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0
        else :
            audio_embeds = torch.zeros((input_ids.shape[0], 1))
            audio_token_mask = torch.tensor([False]).reshape((input_ids.shape[0],1,1))

        # convert decoder with kv-cache
        example_inputs = {"input_ids":input_ids,
                          "audio_embeds":audio_embeds,
                          "audio_token_mask":audio_token_mask,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,
                          "beam_idx": self.next_beam_idx}
        self.dec_request.start_async(example_inputs, share_inputs=True)
        self.dec_request.wait()
        logits1 = torch.from_numpy(self.dec_request.get_tensor("logits").data)

        example_inputs = {"input_ids":input_ids,
                          "audio_embeds":audio_embeds,
                          "audio_token_mask":audio_token_mask,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,
                          "past_key_values": past_key_values}
        logits, past_key_values = self.dec_wrapper(**example_inputs)

        output = CausalLMOutputWithPast(logits=logits1, past_key_values=from_legacy_cache(past_key_values))
        return output

    def convert_config_to_ov(self):
        # Save model config.json
        self.config.save_pretrained(self.ov_config_path.parent)
        
        # Save model generation_config.json
        self.generation_config.save_pretrained(self.ov_config_path.parent)
        
        # Save model processor_config.json
        self.processor.save_pretrained(self.ov_config_path.parent)
        
        # Save model processor for Transformers v4
        self.processor.save_pretrained(self.ov_config_path.parent / "v4")

        # Save model tokenizer
        ### Update tokenizer special tokens 
        import json
        tokenizer_config_file = self.ov_config_path.parent / "v4/tokenizer_config.json"
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            tokenizer_config_init_kwargs = json.load(tokenizer_config_handle)

        extra_special_tokens = tokenizer_config_init_kwargs.pop("extra_special_tokens", ())
        extra_special_tokens_dict = {}
        for extra_special_token in extra_special_tokens:
            extra_special_tokens_dict[extra_special_token] = extra_special_token
        tokenizer_config_init_kwargs["extra_special_tokens"] = extra_special_tokens_dict
        tokenizer_config_init_kwargs["tokenizer_class"] = 'Qwen2TokenizerFast'
        
        
        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config_init_kwargs, f, ensure_ascii=False, indent=2)

        import yaml      
        ov_config_data = {"main_input_name" : self.main_input_name}
        with open(self.ov_config_path, "w") as f:
            yaml.safe_dump(ov_config_data, f)

    def convert_encoder_to_ov(self, input_ids, input_features, input_features_mask):
        if input_features is not None:
            example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
            if not self.ov_encoder_path.exists():
                ov_model = convert_model(self.enc_wrapper, example_input=example_inputs)
                save_model(ov_model, self.ov_encoder_path, compress_to_fp16=False)
                print(f"✅ ModelEncoder completed {self.ov_encoder_path}")
                del ov_model
                cleanup_torchscript_cache()
            audio_embeds = self.enc_wrapper(**example_inputs)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
        else :
            audio_embeds = torch.zeros((input_ids.shape[0], 1))
            audio_token_mask = torch.tensor([False]).reshape((input_ids.shape[0],1,1))
        return audio_embeds, audio_token_mask
    
    def convert_decoder_to_ov(self, input_ids, audio_embeds, audio_token_mask, attention_mask,
                              position_ids, cache_position, past_key_values, 
                              labels = None, use_cache = True, logits_to_keep = 1,
                              quantization_config = None, **kwargs):
        # convert decoder with kv-cache
        example_inputs = {"input_ids":input_ids,
                          "audio_embeds" : audio_embeds,
                          "audio_token_mask": audio_token_mask,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,}
        if not self.ov_decoder_path.exists() and past_key_values is not None:
            example_ov_inputs = example_inputs.copy()
            cache_size = len(past_key_values)
            input_names = ["input_ids",
                           "audio_embeds",
                           "audio_token_mask",
                           "attention_mask",
                           "position_ids",
                           "cache_position",]
            output_names = ["logits"]
            if isinstance(past_key_values, DynamicCache):
                past_key_values = to_legacy_cache(past_key_values)
            for i, cache in enumerate(past_key_values):
                input_names.extend([f"key_values.{i}.key", f"key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_ov_inputs['past_key_values'] = past_key_values

            with torch.no_grad():
                ov_model = ov.convert_model(self.dec_wrapper, example_input=example_ov_inputs)
            
            patch_model_stateful(ov_model, input_names, output_names)
            print("✅ ModelDecoder model successfully converted")

            if quantization_config is not None and "llm" in quantization_config:
                print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
                print("✅ Weights compression finished")
            else:
                ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
            
            ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {self.ov_decoder_path}")

        example_inputs['past_key_values'] = past_key_values
        logits, past_key_values = self.dec_wrapper(**example_inputs)
        output = CausalLMOutputWithPast(logits=logits, past_key_values=from_legacy_cache(past_key_values))
        return output
    
    def convert_to_ov(self, input_ids, input_features, input_features_mask, attention_mask, position_ids,
                      past_key_values, inputs_embeds, labels = None, use_cache = True, cache_position = None,
                      logits_to_keep = 1, quantization_config = None, **kwargs):       
        audio_embeds, audio_token_mask = self.convert_encoder_to_ov(input_ids, input_features, input_features_mask)
        output = self.convert_decoder_to_ov(input_ids=input_ids, 
                                           audio_embeds=audio_embeds,
                                           audio_token_mask=audio_token_mask,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           cache_position=cache_position,
                                           past_key_values=past_key_values,
                                           labels=labels,
                                           use_cache=use_cache,
                                           logits_to_keep=logits_to_keep,
                                           quantization_config=quantization_config,
                                           **kwargs)
        return output

class GlmAsrForOVConvertWrapper1(GenerationMixin):
    _is_stateful = True   # or False

    def __init__(self, model, processor, ov_model_path):
        super().__init__()
        model.config._attn_implementation = "eager"
        self.processor = processor 
        self.config = model.config
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.device = torch.device("cpu")
        
        class ModelAudioEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, input_features, input_features_mask):
                with torch.no_grad():
                    audio_embeds = self.model.get_audio_features(input_features, input_features_mask)
                return audio_embeds

        class ModelInputEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, input_ids):
                with torch.no_grad():
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                    return inputs_embeds

        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, inputs_embeds, attention_mask, position_ids, cache_position, past_key_values):
                with torch.no_grad():
                    if isinstance(past_key_values, list) or isinstance(past_key_values, tuple):
                        # past_key_values = DynamicCache(ddp_cache_data=past_key_values)
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                    result = self.model.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        labels=None,
                        use_cache=True,
                        cache_position=cache_position,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    return result.logits, result.past_key_values.to_legacy_cache()

        self.audio_enc_wrapper = ModelAudioEncoderWrapper(model)
        self.audio_enc_wrapper.eval()

        self.input_enc_wrapper = ModelInputEncoderWrapper(model)
        self.input_enc_wrapper.eval()

        self.dec_wrapper = ModelDecoderWrapper(model)
        self.dec_wrapper.eval()

        self.using_ov = False
        self.ov_core = None
        self.cache_size = 1000
        self.enc_type = 'bf16'
        self.dec_type = 'bf16'

        self.init_model_path(Path(ov_model_path))
        self.load_ov_model()

    def init_model_path(self, ov_path):
        self.ov_audio_encoder_path = ov_path  / GLMASR_Audio_Encoder_MODEL_NAME
        self.ov_input_encoder_path = ov_path / GLMASR_Input_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path  / GLMASR_Decoder_MODEL_NAME
        self.ov_config_path = ov_path  / GLMASR_OV_CONFIG_NAME
        if not self.ov_audio_encoder_path.exists() or not self.ov_input_encoder_path.exists() or not self.ov_decoder_path.exists() or not self.ov_config_path.exists():
            self.converted_to_ov = True
        
    def load_ov_model(self):
        try:
            if self.config is None  :
                self.config = AutoConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            if self.generation_config is None  :
                self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
        except Exception as e:
            print(f"### {e}")
        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]
            
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
           
            device = "CPU"
            ov_config = {}
            ov_config['NUM_STREAMS'] = 1
            ov_config['PERF_COUNT'] = 'NO'
            ov_config['INFERENCE_PRECISION_HINT'] = self.enc_type
            ov_config['PERFORMANCE_HINT'] = 'LATENCY'

            model = self.ov_core.read_model(self.ov_audio_encoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.audio_enc_request = compiled_model.create_infer_request()

            model = self.ov_core.read_model(self.ov_input_encoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.input_enc_request = compiled_model.create_infer_request()

            ov_config['INFERENCE_PRECISION_HINT'] = self.dec_type
            model = self.ov_core.read_model(self.ov_decoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.dec_request = compiled_model.create_infer_request()

            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_audio_encoder_path} or {self.ov_input_encoder_path} or {self.ov_decoder_path} or {self.ov_config_path} failed, {e}")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs
   
    def __call__(self, *args, **kwargs):
        return self.forward(**kwargs)

    def forward(self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        return self.convert_to_ov(input_ids=input_ids,
                                  input_features=input_features,
                                  input_features_mask=input_features_mask,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  cache_position=cache_position,
                                  logits_to_keep=logits_to_keep,
                                  **kwargs)

    def forward_ov(self, 
                      input_ids: torch.LongTensor | None = None,
                      input_features: torch.FloatTensor | None = None,
                      input_features_mask: torch.Tensor | None = None,
                      attention_mask: torch.Tensor | None = None,
                      position_ids: torch.LongTensor | None = None,
                      past_key_values = None,
                      inputs_embeds: torch.FloatTensor | None = None,
                      labels: torch.LongTensor | None = None,
                      use_cache: bool | None = None,
                      cache_position: torch.LongTensor | None = None,
                      logits_to_keep: int | torch.Tensor = 0,
                      **kwargs):
        
        if input_features is not None :
            example_inputs = {"input_ids":input_ids, "input_features":input_features, "input_features_mask":input_features_mask}
            inputs_embeds = self.enc0_wrapper(**example_inputs)
        else :
            example_inputs = {"input_ids":input_ids}
            inputs_embeds = self.enc1_wrapper(**example_inputs)

        # convert decoder with kv-cache
        example_inputs = {"inputs_embeds":inputs_embeds,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,}

        example_inputs['past_key_values'] = past_key_values
        logits, past_key_values = self.dec_wrapper(**example_inputs)
        output = CausalLMOutputWithPast(logits=logits, past_key_values=DynamicCache.from_legacy_cache(past_key_values))
        return output

    def convert_audio_emb_to_ov(self, input_features, input_features_mask):       
        example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
        if not self.ov_audio_encoder_path.exists():
            ov_model = convert_model(self.audio_enc_wrapper, example_input=example_inputs)
            save_model(ov_model, self.ov_audio_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelAudioEncoder completed {self.ov_audio_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()
        audio_embeds = self.audio_enc_wrapper(**example_inputs)
        return audio_embeds

    def convert_inputs_emb_to_ov(self, input_ids):
        if not self.ov_config_path.exists():
            self.config.save_pretrained(self.ov_config_path.parent)
            self.generation_config.save_pretrained(self.ov_config_path.parent)
            self.processor.save_pretrained(self.ov_config_path.parent)
            ov_config_data = {"main_input_name" : self.main_input_name}
            import yaml      
            with open(self.ov_config_path, "w") as f:
                yaml.safe_dump(ov_config_data, f)
  
        example_inputs = {"input_ids":input_ids}
        if not self.ov_input_encoder_path.exists():
            ov_model = convert_model(self.input_enc_wrapper, example_input=example_inputs)
            save_model(ov_model, self.ov_input_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelInputsEncoder completed {self.ov_input_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()
        inputs_embeds = self.input_enc_wrapper(**example_inputs)
        return inputs_embeds
    
    def convert_decoder_to_ov(self, inputs_embeds, attention_mask, position_ids, cache_position, past_key_values,
                      labels = None, use_cache = True, logits_to_keep = 1, quantization_config = None, **kwargs):
        # convert decoder with kv-cache
        example_inputs = {"inputs_embeds":inputs_embeds,
                          "attention_mask":attention_mask,
                          "position_ids":position_ids,
                          "cache_position":cache_position,}
        if not self.ov_decoder_path.exists() and past_key_values is not None:
            example_ov_inputs = example_inputs.copy()
            cache_size = len(past_key_values)
            input_names = ["inputs_embeds",
                           "attention_mask",
                           "position_ids",
                           "cache_position",]
            output_names = ["logits"]
            if isinstance(past_key_values, DynamicCache):
                past_key_values = past_key_values.to_legacy_cache()
            for i, cache in enumerate(past_key_values):
                input_names.extend([f"key_values.{i}.key", f"key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_ov_inputs['past_key_values'] = past_key_values
            
            with torch.no_grad():
                ov_model = ov.convert_model(self.dec_wrapper, example_input=example_ov_inputs)
            
            patch_model_stateful(ov_model, input_names, output_names)
            print("✅ ModelDecoder model successfully converted")

            if quantization_config is not None and "llm" in quantization_config:
                print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
                print("✅ Weights compression finished")
            else:
                ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
            
            ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {self.ov_decoder_path}")

        example_inputs['past_key_values'] = past_key_values
        logits, past_key_values = self.dec_wrapper(**example_inputs)
        output = CausalLMOutputWithPast(logits=logits, past_key_values=DynamicCache.from_legacy_cache(past_key_values))
        return output

    def convert_to_ov(self,
                      input_ids: torch.LongTensor | None = None,
                      input_features: torch.FloatTensor | None = None,
                      input_features_mask: torch.Tensor | None = None,
                      attention_mask: torch.Tensor | None = None,
                      position_ids: torch.LongTensor | None = None,
                      past_key_values = None,
                      inputs_embeds: torch.FloatTensor | None = None,
                      labels: torch.LongTensor | None = None,
                      use_cache: bool | None = None,
                      cache_position: torch.LongTensor | None = None,
                      logits_to_keep: int | torch.Tensor = 0,
                      quantization_config = None,
                      **kwargs):      
        inputs_embeds = self.convert_inputs_emb_to_ov(input_ids)
        
        if input_features is not None :
            audio_embeds = self.convert_audio_emb_to_ov(input_features, input_features_mask)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
                )

        output = self.convert_decoder_to_ov(inputs_embeds=inputs_embeds,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           cache_position=cache_position,
                                           past_key_values=past_key_values,
                                           labels=labels,
                                           use_cache=use_cache,
                                           logits_to_keep=logits_to_keep,
                                           quantization_config=quantization_config,
                                           **kwargs)
        return output

FUNASR_Audio_Encoder_MODEL_NAME = "funasr_audio_encoder.xml"
FUNASR_Audio_Encoder_CTC_MODEL_NAME = "funasr_audio_encoder_ctc.xml"
FUNASR_Input_Encoder_MODEL_NAME = "funasr_input_encoder.xml"
FUNASR_Decoder_MODEL_NAME = "funasr_llm_decoder.xml"
FUNASR_OV_CONFIG_NAME = "ov_config.yaml"
FUNASR_Frontend_CONFIG_NAME = "frontend_config.json"

def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    items = []
    try:
        # The current version only supports batch_size==1.
        log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
        assert log_probs.shape[1] >= targets.shape[1]
        alignments, scores = F.forced_align(log_probs, targets, blank=blank)
        alignments, scores = alignments[0], torch.exp(scores[0]).tolist()
        # use enumerate to keep track of the original indices, then group by token value
        for token, group in groupby(enumerate(alignments), key=lambda item: item[1]):
            if token == blank:
                continue
            group = list(group)
            start = group[0][0]
            end = start + len(group)
            score = max(scores[start:end])
            items.append(
                {
                    "token": token.item(),
                    "start_time": start,
                    "end_time": end,
                    "score": round(score, 3),
                }
            )
    except:
        pass
    return items

class FunAsrNanoConverterWrapper(GenerationMixin) :
    _is_stateful = True
    
    def __init__(self, model, kwargs, ov_model_path):
        class ModelAudioEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.float().eval()
                self.model.audio_encoder = model.audio_encoder.float().eval()
                self.model.audio_adaptor = model.audio_adaptor.float().eval()

            def forward(self, speech, speech_lengths):
                with torch.no_grad():
                    encoder_out, encoder_out_lens = self.model.audio_encoder(speech, speech_lengths)
                    adaptor_out, adaptor_out_lens = self.model.audio_adaptor(encoder_out, encoder_out_lens)
                return adaptor_out, adaptor_out_lens

        class ModelAudioEncoderWithCTCWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.float().eval()
                self.model.audio_encoder = model.audio_encoder.float().eval()
                self.model.audio_adaptor = model.audio_adaptor.float().eval()
                self.ctc_decoder = model.ctc_decoder.float().eval()
                self.ctc = model.ctc.float().eval()

            def forward(self, speech, speech_lengths):
                with torch.no_grad():
                    # encoder_out, encoder_out_lens = self.model.encode(speech, speech_lengths)
                    encoder_out, encoder_out_lens = self.model.audio_encoder(speech, speech_lengths)
                    adaptor_out, adaptor_out_lens = self.model.audio_adaptor(encoder_out, encoder_out_lens)
                    decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
                    ctc_logits = self.ctc.log_softmax(decoder_out)
                    yseqs = ctc_logits.argmax(dim=-1)
                return adaptor_out, adaptor_out_lens, ctc_logits, yseqs

        class ModelTextEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()
                self.model.llm = self.model.llm.float().eval()
                self.model.llm.model = self.model.llm.model.float().eval()

            def forward(self, input_ids):
                with torch.no_grad():
                    inputs_embeds = self.model.llm.model.get_input_embeddings()(input_ids)
                return inputs_embeds

        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()
                self.model.llm = self.model.llm.float().eval()
                self.model.llm.model = self.model.llm.model.float().eval()

            # def forward(self, inputs_embeds, attention_mask, position_ids, cache_position, past_key_values):
            def forward(self, inputs_embeds, attention_mask, past_key_values):
                with torch.no_grad():
                    if isinstance(past_key_values, list) or isinstance(past_key_values, tuple):
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                    result = self.model.llm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        # position_ids=position_ids,
                        # cache_position=cache_position,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                    return result.logits, result.past_key_values.to_legacy_cache()

        self.ov_model_path = Path(ov_model_path)
        self.ov_audio_path = self.ov_model_path  / FUNASR_Audio_Encoder_MODEL_NAME
        self.ov_audio_ctc_path = self.ov_model_path  / FUNASR_Audio_Encoder_CTC_MODEL_NAME
        self.ov_text_path = self.ov_model_path  / FUNASR_Input_Encoder_MODEL_NAME
        self.ov_decoder_path = self.ov_model_path  / FUNASR_Decoder_MODEL_NAME
        self.ov_config_path = self.ov_model_path  / FUNASR_OV_CONFIG_NAME
        self.frontend_config_path = self.ov_model_path  / FUNASR_Frontend_CONFIG_NAME

        super().__init__()
        model.llm.config._attn_implementation = "eager"
        self.config = model.llm.config
        self.generation_config = model.llm.generation_config
        self.main_input_name = model.llm.main_input_name
        self.device = torch.device("cpu")

        self.pt_model = model
        self.config = model.llm.config
        self.generation_config = model.llm.generation_config
        self.use_low_frame_rate = model.use_low_frame_rate
        self.pad_token_id = model.llm.config.pad_token_id or model.llm.config.eos_token_id
        self.blank_id = model.blank_id
        self.using_ctc = True if model.ctc_decoder is not None else False
        # self.using_ctc = False
        if self.using_ctc:
            self.ctc_tokenizer = model.ctc_tokenizer
            self.ctc_tokenizer_str = (
                kwargs.get("ctc_tokenizer", None)
                if "ctc_tokenizer" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer"]
            )
            self.ctc_tokenizer_conf = (
                kwargs.get("ctc_tokenizer_conf", None)
                if "ctc_tokenizer_conf" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
            )
            self.ctc = model.ctc

        self.audio_enc_wrapper = ModelAudioEncoderWrapper(model)
        self.audio_enc_wrapper.eval()
        self.audio_enc_ctc_wrapper = ModelAudioEncoderWithCTCWrapper(model)
        self.audio_enc_ctc_wrapper.eval()
        self.text_enc_wrapper = ModelTextEncoderWrapper(model)
        self.text_enc_wrapper.eval()
        self.dec_wrapper = ModelDecoderWrapper(model)
        self.dec_wrapper.eval()

    def get_prompt_did(self):
        prompt = f"语种方言识别："
        return prompt

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        prompt = self.get_prompt_did()
        data_in = [self.pt_model.generate_chatml(prompt, data) for data in data_in]

        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append("rand_key_" + "".join(random.choice(chars) for _ in range(13)))

        meta_data = {}
        contents = self.pt_model.data_template(data_in[0])
        output = self.pt_model.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        self.convert_ov_others(tokenizer, frontend, **kwargs)

        speech = output['speech']
        speech_lengths = output['speech_lengths'][:, 0]
        input_ids = output['source_ids']
        fake_token_len = output['fake_token_len']
        fbank_beg = output['fbank_beg']
        attention_mask = output["attention_mask"]

        input_ids[input_ids < 0] = 0
        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        adaptor_out, adaptor_out_lens, inputs_embeds, ctc_logits, yseqs = self.convert_ov_encoder_model(speech, speech_lengths, input_ids)

        batch_size, token_num, dims = inputs_embeds.shape
        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = adaptor_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        print(f"#### e={e}")
                        speech_token_len = adaptor_out_lens[speech_idx].item()
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1

        self.pt_model.llm = self.pt_model.llm.to(torch.float32)
        inputs_embeds = inputs_embeds.to(torch.float32)  
        llm_kwargs = kwargs.get("llm_kwargs", {})
        generated_ids = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_length", 512),
            pad_token_id=self.pad_token_id,
            **llm_kwargs,
        )

        ctc_results = []
        
        if self.using_ctc:
            b, _, _ = ctc_logits.size()
            if isinstance(key[0], (list, tuple)):
                key = key[0]
            if len(key) < b:
                key = key * b
            for i in range(b):
                x = ctc_logits[i, :, :]
                # yseq = x.argmax(dim=-1)
                yseq = yseqs[i, :]
                yseq = torch.unique_consecutive(yseq, dim=-1)
                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                # Change integer-ids to tokens
                text = self.ctc_tokenizer.decode(token_int)
                ctc_results.append({"key": key[i], "text": text, "ctc_logits": x})

        label = contents["assistant"][-1]
        response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
        )[0]
        loss = None
        response = kwargs.get("prev_text", "") + response

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": re.sub(r"\s+", " ", response.replace("/sil", " ")),
            "text_tn": response_clean,
            "label": label,
        }
        results.append(result_i)

        for ctc_result, result in zip(ctc_results, results):
            result["ctc_text"] = ctc_result["text"].replace("<|nospeech|>", "")
            target_ids = torch.tensor(
                self.ctc_tokenizer.encode(result["ctc_text"]), dtype=torch.int64
            )
            result["ctc_timestamps"] = forced_align(
                ctc_result["ctc_logits"], target_ids, self.blank_id
            )
            target_ids = torch.tensor(self.ctc_tokenizer.encode(result["text"]), dtype=torch.int64)
            result["timestamps"] = forced_align(ctc_result["ctc_logits"], target_ids, self.blank_id)
            for timestamps in [result["timestamps"], result["ctc_timestamps"]]:
                for timestamp in timestamps:
                    timestamp["token"] = self.ctc_tokenizer.decode([timestamp["token"]])
                    timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                    timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000

        return results, meta_data
      
    def save_frontend_config(self, frontend, **kwargs):
        if not self.frontend_config_path.exists() :
            print(f"### frontend={frontend.__class__.__name__}")
            frontend_config = {
                # Frontend settings
                "frontend_type": "WavFrontend",
                "cmvn_file": frontend.cmvn_file,
                "fs": frontend.fs,
                "window": frontend.window,
                "n_mels": frontend.n_mels,
                "frame_length": frontend.frame_length,
                "frame_shift": frontend.frame_shift,
                "filter_length_min": frontend.filter_length_min,
                "filter_length_max": frontend.filter_length_max,
                "lfr_m": frontend.lfr_m,
                "lfr_n": frontend.lfr_n,
                "dither": 0.0, # Set to 0 for deterministic inference (original uses dither=1.0 which adds random noise)
                "snip_edges": frontend.snip_edges,
                "upsacle_samples": frontend.upsacle_samples,
            }
            with open(self.frontend_config_path, "w") as f:
                json.dump(frontend_config, f, indent=2)
            print("✅ Frontend config exported")

    def convert_ov_others(self, tokenizer, frontend, **kwargs):
        #save llm config
        self.config.save_pretrained(self.ov_config_path.parent)

        #save llm generation_config
        self.generation_config.save_pretrained(self.ov_config_path.parent)

        #save tokenizer
        tokenizer.save_pretrained(self.ov_config_path.parent)

        #save frontend config        
        self.save_frontend_config(frontend, **kwargs)
        
        ov_config_data = {"main_input_name" : self.main_input_name,
                          "use_low_frame_rate" : self.use_low_frame_rate,
                          "pad_token_id" : self.pad_token_id,
                          "using_ctc": self.using_ctc}

        #save ctc tokenizer
        if self.using_ctc:
            vocab_path = Path(self.ctc_tokenizer_conf['vocab_path'])
            shutil.copy(vocab_path , self.ov_config_path.parent)
            new_ctc_tokenizer_conf = self.ctc_tokenizer_conf.copy()
            new_ctc_tokenizer_conf['vocab_path'] = vocab_path.name
            ov_config_data['ctc_tokenizer_conf'] = new_ctc_tokenizer_conf
            ov_config_data['ctc_tokenizer'] = self.ctc_tokenizer_str
            ov_config_data['blank_id'] = self.blank_id

        #save ov config
        if not self.ov_config_path.exists() :
            with open(self.ov_config_path, "w") as f:
                yaml.safe_dump(ov_config_data, f)
            print("✅ OV config exported")

    def convert_ov_encoder_model(self, speech, speech_lengths, input_ids, **kwargs):
        if not self.ov_audio_path.exists() :
            example_inputs = {"speech":speech, "speech_lengths":torch.tensor([1]).to(dtype=torch.int32)}
            ov_model = convert_model(self.audio_enc_wrapper, example_input=example_inputs)
            save_model(ov_model, self.ov_audio_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_audio_path}")
            del ov_model
            cleanup_torchscript_cache()

        if not self.ov_audio_ctc_path.exists() :
            example_inputs = {"speech":speech, "speech_lengths":torch.tensor([1]).to(dtype=torch.int32)}
            ov_model = convert_model(self.audio_enc_ctc_wrapper, example_input=example_inputs)
            save_model(ov_model, self.ov_audio_ctc_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_audio_ctc_path}")
            del ov_model
            cleanup_torchscript_cache()

        adaptor_out, adaptor_out_lens = self.audio_enc_wrapper(speech, speech_lengths)
    
        adaptor_out, adaptor_out_lens, ctc_logits, yseqs = self.audio_enc_ctc_wrapper(speech, speech_lengths)

        if not self.ov_text_path.exists() :
            example_inputs = {"input_ids":input_ids}
            ov_model = convert_model(self.text_enc_wrapper, example_input=example_inputs)
            save_model(ov_model, self.ov_text_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_text_path}")
            del ov_model
            cleanup_torchscript_cache()

        inputs_embeds = self.text_enc_wrapper(input_ids)

        return adaptor_out, adaptor_out_lens, inputs_embeds, ctc_logits, yseqs

    def __call__(self, *args, **kwargs):
        return self.forward(**kwargs)
    
    def forward(self, inputs_embeds=None, attention_mask=None, past_key_values=None, 
                input_ids=None, position_ids=None, cache_position=None,
                use_cache=False, return_dict=True, **kwargs):
        with torch.no_grad():
            if inputs_embeds is None :
                inputs_embeds = self.text_enc_wrapper(input_ids)
                self.convert_ov_decoder_model(inputs_embeds=inputs_embeds,
                                              attention_mask=attention_mask,
                                            #   position_ids=position_ids,
                                            #   cache_position=cache_position,
                                              past_key_values=past_key_values)
            logits, past_key_values = self.dec_wrapper(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                # cache_position=cache_position,
                past_key_values=past_key_values,
            )
            outputs = CausalLMOutputWithPast(logits=logits, 
                        past_key_values=DynamicCache.from_legacy_cache(past_key_values))
            return outputs

    def convert_ov_decoder_model(self, inputs_embeds, attention_mask,
                                #  position_ids, cache_position,
                                 past_key_values,
                                 quantization_config=None):
        if not self.ov_decoder_path.exists() :
            caches = []
            input_names = ["inputs_embeds", "attention_mask"]#, "position_ids", "cache_position"]
            output_names = ["logits"]

            if isinstance(past_key_values, DynamicCache):
                past_key_values = past_key_values.to_legacy_cache()
            for i, cache in enumerate(past_key_values):
                input_names.extend([f"key_values.{i}.key", f"key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_input = {"inputs_embeds":inputs_embeds,
                             "attention_mask": attention_mask,
                            #  "position_ids": position_ids,
                            #  "cache_position": cache_position,
                             "past_key_values": past_key_values}
            ov_model = ov.convert_model(self.dec_wrapper, example_input=example_input)
            
            patch_model_stateful(ov_model, input_names, output_names)

            print("✅ ModelDecoder model successfully converted")

            if quantization_config is not None and "llm" in quantization_config:
                print(f"⌛ Weights compression with {quantization_config['llm']['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config["llm"])
                print("✅ Weights compression finished")
            else:
                ov_model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
            
            ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {self.ov_decoder_path}")
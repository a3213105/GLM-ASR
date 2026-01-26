# -*- coding: UTF-8 -*-
import gc
import numpy as np
from pathlib import Path
import os
import json

import torch
import torch.nn.functional as F

from transformers import AutoConfig, DynamicCache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
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

def patch_model_stateful(ov_model, input_names, output_names, input_batch_name="input_ids"):
    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})
    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})
    patch_stateful(ov_model, input_batch_name)
    return ov_model

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

# def to_legacy_cache(caches):
#     """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
#     legacy_cache = ()
#     if transformers_ver > version.parse("4"):
#         for layer_idx in range(len(caches)):
#             legacy_cache += ((caches[layer_idx][0], caches[layer_idx][1]),)
#     else :
#         for layer_idx in range(len(caches)):
#             legacy_cache += ((caches.key_cache[layer_idx], caches.value_cache[layer_idx]),)
#     return legacy_cache

RTOL_STRICT = 1e-3
ATOL_STRICT = 1e-3
equal_nan=True

# from transformers.generation import GenerationMixin
class GlmAsrForOVConvertWrapper(GenerationMixin):
    _is_stateful = True   # or False
    _keep_in_fp32_modules_strict = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, model, processor, ov_model_path):
        super().__init__()
        model.config._attn_implementation = "eager"
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
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

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
                    # past_key_values = to_legacy_cache(pkv)
                    past_key_values = result.past_key_values.to_legacy_cache()
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
        # print(f"### labels={labels}, logits_to_keep={logits_to_keep}, use_cache={use_cache}")
        if input_features is not None:
            example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
            audio_embeds1 = self.enc_wrapper(**example_inputs)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)

            self.enc_request.start_async(example_inputs, share_inputs=True)
            self.enc_request.wait()
            audio_embeds = torch.from_numpy(self.enc_request.get_output_tensor(0).data)
            # if torch.equal(audio_embeds, audio_embeds1) :
            if torch.allclose(audio_embeds, audio_embeds1, rtol=RTOL_STRICT, atol=ATOL_STRICT, equal_nan=equal_nan) :
                print(f"✅ encoder output match, {cache_position.max()}")
            else :
                print(f"❌ encoder output not match, {cache_position.max()}")
                print(f"audio_embeds={audio_embeds.shape}, audio_embeds1={audio_embeds1.shape}")
                mask = ~torch.isclose(audio_embeds, audio_embeds1, rtol=RTOL_STRICT, atol=ATOL_STRICT, equal_nan=equal_nan)
                diff_count = mask.sum()
                print(f"total_diff={diff_count}")
                row_diff = mask.any(dim=-1)
                # print(f"row_diff={row_diff}")

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
        if torch.allclose(logits, logits1, rtol=RTOL_STRICT, atol=ATOL_STRICT, equal_nan=equal_nan) :
            print(f"✅ decoder output match, {cache_position.max()}")
        else :
            print(f"❌ decoder output not match, {cache_position.max()}")
            # print(f"logits={logits.shape}, logits1={logits1.shape}")
            mask = ~torch.isclose(logits, logits1, rtol=RTOL_STRICT, atol=ATOL_STRICT, equal_nan=equal_nan)
            diff_count = mask.sum()
            print(f"total_diff={diff_count}")
            row_diff = mask.any(dim=-1)
            # print(f"row_diff={row_diff}")         
            diff_positions = torch.nonzero(mask)
            # print(f"diff_positions={diff_positions}")
  

        output = CausalLMOutputWithPast(logits=logits1, past_key_values=DynamicCache.from_legacy_cache(past_key_values))
        return output

    def convert_encoder_to_ov(self, input_ids, input_features, input_features_mask):
        if not self.ov_config_path.exists():
            self.config.save_pretrained(self.ov_config_path.parent)
            self.generation_config.save_pretrained(self.ov_config_path.parent)
            self.processor.save_pretrained(self.ov_config_path.parent)
            ov_config_data = {"main_input_name" : self.main_input_name}
            import yaml      
            with open(self.ov_config_path, "w") as f:
                yaml.safe_dump(ov_config_data, f)

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
                past_key_values = past_key_values.to_legacy_cache()
            for i, cache in enumerate(past_key_values):
                input_names.extend([f"key_values.{i}.key", f"key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_ov_inputs['past_key_values'] = past_key_values
            
            with torch.no_grad():
                ov_model = ov.convert_model(self.dec_wrapper, example_input=example_ov_inputs)
            
            patch_model_stateful(ov_model, input_names, output_names, "input_ids")
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
        print(f"inputs_embeds({inputs_embeds.shape})={inputs_embeds.float()}")
        print(f"logits({logits.shape})={logits.float()}")
        # output = CausalLMOutputWithPast(logits=logits, past_key_values=DynamicCache(ddp_cache_data=past_key_values))
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
            
            patch_model_stateful(ov_model, input_names, output_names, "inputs_embeds")
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
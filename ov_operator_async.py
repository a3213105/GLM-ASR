from array import array
from locale import ABDAY_1
import numpy as np
from datetime import datetime
from openvino import Core,Model, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
import os
import copy
from pathlib import Path
from typing import Optional, Tuple, Callable, Any, Union
import json
import string
import random
import re
import types
import time

from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

class OV_Operator(object):
    core = None
    model = None
    model_dynamic = None
    input_names = None
    input_shapes = None
    out_name = None
    exec_net = None
    infer_queue = None
    request = None
    outputs= None

    def __init__(self, model, core=None, postprocess=None):
        self.postprocess = postprocess
        if core is None :
            self.core = Core()
        else :
            self.core = core
        self.model = self.core.read_model(model=model)
        output_size = self.model.get_output_size()
        self.outputs = []
        for i in range (0,output_size):
            self.outputs.append(i)
        # print('output: {}'.format(len(self.outputs)))
        self.input_names = []
        self.input_shapes = []
        ops = self.model.get_ordered_ops()
        for it in ops:
            if it.get_type_name() == 'Parameter':
                self.input_names.append(it.get_friendly_name())
                self.input_shapes.append(it.partial_shape)
                # print('input {}: {}'.format(it.get_friendly_name(),it.partial_shape))
        self.input_name = self.input_names[0]
        
    # def __init__(self, model, stream_num, bf16=True, f16=False,
    #              core=None, shape=None, postprocess=None, **kwargs):
    #     self.postprocess = postprocess
    #     if core is None :
    #         self.core = Core()
    #     else :
    #         self.core = core
    #     self.model = self.core.read_model(model=model)
    #     output_size = self.model.get_output_size()
    #     self.outputs = []
    #     for i in range (0,output_size):
    #         self.outputs.append(i)
    #     # print('output: {}'.format(len(self.outputs)))
    #     self.input_names = []
    #     self.input_shapes = []
    #     ops = self.model.get_ordered_ops()
    #     for it in ops:
    #         if it.get_type_name() == 'Parameter':
    #             self.input_names.append(it.get_friendly_name())
    #             self.input_shapes.append(it.partial_shape)
    #             # print('input {}: {}'.format(it.get_friendly_name(),it.partial_shape))
    #     self.input_name = self.input_names[0]
    #     self.setup_model(stream_num, bf16=bf16, f16=f16, shape=shape, **kwargs)

    def create_single_request(self, bf16) :
        config = self.prepare_for_cpu(1, bf16)
        if self.model_dynamic is not None:
            self.exec_net_single = self.core.compile_model(self.model_dynamic, 'CPU', config)
        else :            
            self.exec_net_single = self.core.compile_model(self.model, 'CPU', config)
        self.request = self.exec_net_single.create_infer_request()

    def setup_model(self, stream_num, bf16=True, f16=False, shape=None) :
        if shape is not None :
            self.model.reshape({self.input_name: shape})
        config = self.prepare_for_cpu(stream_num, bf16, f16)
        self.exec_net = self.core.compile_model(self.model, 'CPU', config)
        self.num_requests = self.exec_net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
 
        if self.num_requests > 1:
            self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
            self.create_single_request(bf16)
        else :
            self.request = self.exec_net.create_infer_request()
            self.infer_queue = None
        # self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
        # self.create_single_request(bf16)
        # print('Model ({})  using {} streams'.format(self.model.get_friendly_name(), self.num_requests))

    def prepare_for_cpu(self, stream_num, bf16=True, f16=False) :
        device = "CPU"
        hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
        data_type = 'bf16' if bf16 else 'f16' if f16 else 'f32'
        config = {}
        supported_properties = self.core.get_property(device, 'SUPPORTED_PROPERTIES')
        config['NUM_STREAMS'] = str(stream_num)
        config['PERF_COUNT'] = 'NO'
        config['INFERENCE_PRECISION_HINT'] = data_type #'bf16'#'f32'
        config['PERFORMANCE_HINT'] = hint # 'THROUGHPUT' #"LATENCY"
        # print(f"OV_Operator prepare_for_cpu: {config}")
        return config

    def __call__(self, input_tensors):
        nsize=len(input_tensors)
        if self.request and nsize==1:
            self.res.sync_clean()
            for i, input_tensor in enumerate(input_tensors):
                result = self.request.infer(input_tensor, share_inputs=True)
                self.res.sync_parser(result, i)
        elif self.infer_queue :
            for i, input_tensor in enumerate(input_tensors):
                self.infer_queue.start_async(input_tensor, userdata=i, share_inputs=True)
            self.infer_queue.wait_all()
        else :
            print("Can not enter here!!!")
        return nsize

class OV_Result :
    results = None
    outputs = None
    def __init__(self, outputs) :
        self.outputs = outputs
        self.results = {}
        #for i in outputs:
        #    #print('add results item {}'.format(i))
        #    self.results[i] = {}

    def completion_callback(self, infer_request: InferRequest, index: any) :
        #if index not in self.results :
        self.results[index] = []
        for i in self.outputs:
            self.results[index].append(copy.deepcopy(infer_request.get_output_tensor(i).data))
        return 

    def sync_parser(self, result, index: any) :
        self.results[index] = []
        values = result.values()
        for i, value in enumerate(values):
            # print("output {} value shape {}".format(i, value.shape))
            self.results[index].append(value)
        return 
    
    def sync_clean(self):
        self.results = {}

class base_torch_function_ov :
    def eval(self):
        return self
    
    def cpu(self):
        return self

FireRedAsrAed_Encoder_MODEL_NAME = "FireRedASR_AED_encoder_ov.xml"
FireRedAsrAed_Decoder_MODEL_NAME = "FireRedASR_AED_decoder_ov.xml"

class FireRedAsrAedConformerEncoderModel(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=True) :
        super().setup_model(stream_num, bf16, f16)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, inputs):
        output = self.request.infer(inputs)
        return (output[0], output[1])
    
    def start_async(self, inputs):
        self.request.start_async(inputs)
        
    def get_data_async(self):
        self.request.wait()
        encoder_outputs = self.request.get_output_tensor(0).data
        src_masks = self.request.get_output_tensor(1).data
        return (encoder_outputs, src_masks)

class FireRedAsrAedTransformerDecoderModel(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        self.next_beam_idx = None
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 2, bf16=True, f16=True) :
        super().setup_model(stream_num, bf16, f16)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def clear_state(self, B) :
        self.request.reset_state()
        self.next_beam_idx = np.arange(B, dtype=int)
        
    def __call__(self, input_dict):
        input_dict["beam_idx"] = self.next_beam_idx
        output = self.request.infer(input_dict, share_inputs=True)
        return (output[0], output[1], output[2])
        # topB_row_number_in_ys = self.request.get_tensor("topB_row_number_in_ys").data
        # t_ys = self.request.get_tensor("new_t_ys").data
        # scores = self.request.get_tensor("new_scores").data
        # return (topB_row_number_in_ys, t_ys, scores)

class FireRedAsrAedEncDecModel(base_torch_function_ov) :
    def __init__(self, args, ov_core, model_path, enc_type, dec_type, cache_size, ov_version = "ov_model_v0"):
        self.ov_version = ov_version
        self.init(args, ov_core, model_path, enc_type, dec_type, cache_size)
        
    def init(self, args, ov_core, model_path, enc_type, dec_type, cache_size):
        ov_path = Path(model_path)
        self.converted_to_ov = False
        self.using_ov = False
        self.init_model_path(ov_path)
        self.cache_size = cache_size
        self.ov_core = ov_core
        self.enc_type = enc_type
        self.dec_type = dec_type
        if self.enc_type in "f32f16bf16" and self.dec_type in "f32f16bf16" :
            self.load_ov_model()

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path.parent / self.ov_version / FireRedAsrAed_Decoder_MODEL_NAME
        if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists():
            self.converted_to_ov = True
        
    def load_ov_model(self):
        try :
            if self.ov_core is None :
                self.ov_core = Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
           
            self.enc_request = FireRedAsrAedConformerEncoderModel(self.ov_encoder_path, self.ov_core)
            self.enc_request.setup_model(1, True if self.enc_type=='bf16' else False, True if self.enc_type=='f16' else False)
            self.dec_request = FireRedAsrAedTransformerDecoderModel(self.ov_decoder_path, self.ov_core)
            self.dec_request.setup_model(1, True if self.dec_type=='bf16' else False, True if self.dec_type=='f16' else False)

            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")


    def encoder(self, inputs, beam_size) :
        self.enc_request.start_async(inputs)
        self.dec_request.clear_state(beam_size)
        return self.enc_request.get_data_async()

    def decoder(self, inputs) :
        return self.dec_request(inputs)

GLMASR_Audio_Encoder_MODEL_NAME = "glm_asr_audio_encoder.xml"
GLMASR_Input_Encoder_MODEL_NAME = "glm_asr_input_encoder.xml"
GLMASR_Encoder_MODEL_NAME = "glm_asr_encoder.xml"
GLMASR_Decoder_MODEL_NAME = "glm_asr_decoder.xml"
GLMASR_OV_CONFIG_NAME = "ov_config.yaml"

class GLMASREncoderModel(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=True) :
        super().setup_model(stream_num, bf16, f16)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, inputs):
        output = self.request.infer(inputs)
        return output[0]
    
    def start_async(self, inputs):
        self.request.start_async(inputs)
        
    def get_data_async(self):
        self.request.wait()
        return self.request.get_output_tensor(0).data

class GLMASRDecoderModel(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        self.next_beam_idx = None
        self.past_length = 0
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 1, bf16=True, f16=True) :
        super().setup_model(stream_num, bf16, f16)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def clear_state(self, B) :
        self.request.reset_state()
        self.past_length = 0
        self.next_beam_idx = np.arange(B, dtype=int)
        
    def __call__(self, input_dict):
        input_dict["beam_idx"] = self.next_beam_idx
        output = self.request.infer(input_dict, share_inputs=True)
        return output[0]

class GlmAsrEncoderConfig(PretrainedConfig):
    model_type = "glmasr_encoder"

    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=None,
        hidden_act="gelu",
        max_position_embeddings=1500,
        initializer_range=0.02,
        rope_parameters=None,
        attention_dropout=0.0,
        num_mel_bins=128,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.num_mel_bins = num_mel_bins

        kwargs.setdefault("partial_rotary_factor", 0.5)
        super().__init__(**kwargs)

class GlmAsrConfig(PretrainedConfig):
    model_type = "glmasr"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 59264,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "eos_token_id": [59246, 59253, 59255],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=59260,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "glmasr_encoder")
            if audio_config["model_type"] == "glmasr_encoder":
                audio_config = GlmAsrEncoderConfig(**audio_config)
            else :
                audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            # audio_config = CONFIG_MAPPING["glmasr_encoder"]()
            audio_config = GlmAsrEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.audio_token_id = audio_token_id
        self.projector_hidden_act = projector_hidden_act

        super().__init__(**kwargs)

from transformers.processing_utils import ProcessorMixin, ProcessingKwargs
from transformers.audio_utils import make_list_of_audio
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

class GlmAsrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "chunk_length": 30.0,
            "return_attention_mask": True,
            "padding": "max_length",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
    }


class GlmAsrProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]  # ProcessorMixin 需要
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "LlamaTokenizerFast"  # 或者你实际的 tokenizer 类名

    def __init__(
        self,
        feature_extractor,
        model_path,
        audio_token="<|pad|>",
        default_transcription_prompt="Please transcribe this audio into text",
        max_audio_len=655,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        chat_template_file = model_path + "/chat_template.jinja"
        with open(chat_template_file, encoding="utf-8") as f:
            self.chat_template = f.read()

        self.audio_token = audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)
        self.default_transcription_prompt = default_transcription_prompt
        self.max_audio_len = max_audio_len
        self.feature_extractor = feature_extractor

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        merge_factor = 4
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        num_tokens = (audio_lengths - merge_factor) // merge_factor + 1
        return num_tokens

    def __call__(self, text, audio, output_labels = False, **kwargs,):
        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            GlmAsrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_inputs = {}
        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            # Determine number of chunks per sample, and flatten
            window_size = int(audio_kwargs["sampling_rate"] * audio_kwargs["chunk_length"])
            max_windows = int(self.max_audio_len // audio_kwargs["chunk_length"])

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []

            for audio_el in audio:
                n_samples = int(audio_el.shape[0])
                n_win = max(1, (n_samples + window_size - 1) // window_size)
                if n_win > max_windows:
                    logger.warning(
                        f"Audio duration ({n_samples / audio_kwargs['sampling_rate']:.1f}s) exceeds {self.max_audio_len}s; truncating to first {self.max_audio_len}s."
                    )
                    n_win = max_windows
                per_sample_windows.append(n_win)

                time_cap = min(n_samples, n_win * window_size)
                for i in range(n_win):
                    start = i * window_size
                    end = min((i + 1) * window_size, time_cap)
                    flat_chunks.append(audio_el[start:end])

            # Feature extraction
            audio_inputs = self.feature_extractor(flat_chunks, **audio_kwargs)
            padding_mask = audio_inputs.pop("attention_mask")
            audio_inputs["input_features_mask"] = padding_mask

            # Compute sequence lengths token counting
            audio_lengths = torch.stack([s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)])
            audio_tokens_lengths = self._get_audio_token_length(audio_lengths)

            # expand audio tokens in text
            for i, audio_length in enumerate(audio_tokens_lengths):
                expanded = re.sub(re.escape(self.audio_token), self.audio_token * audio_length, text[i])
                text[i] = expanded

        # Tokenize
        text_inputs = self.tokenizer(text, **text_kwargs)

        data = {**text_inputs, **audio_inputs}
        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        fea_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tok_names + fea_names + ["input_features_mask"]))

    def apply_transcription_request(self, audio, prompt=None,**kwargs, ):
        if isinstance(audio, str):
            audio_items: list[str | np.ndarray] = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(el, str) for el in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_item}
                        if isinstance(audio_item, str)
                        else {"type": "audio", "audio": audio_item},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def batch_decode(self, *args, strip_prefix=False, **kwargs):
        decoded = self.tokenizer.batch_decode(*args, **kwargs)
        if strip_prefix:
            decoded = [self._strip_assistant_prefix_and_quotes(text) for text in decoded]
        return decoded

    def _strip_assistant_prefix_and_quotes(self, text: str) -> str:
        stripped = text.strip()

        for prefix in (
            "The spoken content of the audio is",
            "The transcription of the audio is",
        ):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break

        if stripped.endswith("."):
            stripped = stripped[:-1].strip()

        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
            stripped = stripped[1:-1].strip()

        return stripped

class BaseEncDecGenModel(GenerationMixin):
    _is_stateful = True   # or False

    def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size):
        super().__init__()
        self.device = torch.device("cpu")
        self.init(ov_core, model_path, enc_type, dec_type, cache_size)
        
    def init(self, ov_core, model_path, enc_type, dec_type, cache_size):
        ov_path = Path(model_path)
        self.converted_to_ov = False
        self.using_ov = False
        self.init_model_path(ov_path)
        self.cache_size = cache_size
        self.ov_core = ov_core
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.next_beam_idx = None
        self._past_length = None
        if self.enc_type in "f32f16bf16" and self.dec_type in "f32f16bf16" and not self.converted_to_ov:
            self.load_ov_config()
            self.load_ov_model()

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path
        self.ov_decoder_path = ov_path
        self.ov_config_path = ov_path
        raise NotImplementedError()

    def load_ov_config(self):
        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]
        except Exception as e:
            print(f"### ov load {self.ov_config_path} failed, {e}")
        raise NotImplementedError()
    
    def load_ov_model(self):
        try :            
            if self.ov_core is None :
                self.ov_core = Core()
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
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
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
        raise NotImplementedError()
    
    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def _reorder_cache(self, past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        return model_inputs

class GlmAsrEncDecModel(BaseEncDecGenModel) :
    def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size):
        super().__init__(ov_core, model_path, enc_type, dec_type, cache_size)

    def init_model_path(self, ov_path):
        self.ov_encoder_path = ov_path  / GLMASR_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path  / GLMASR_Decoder_MODEL_NAME
        self.ov_config_path = ov_path  / GLMASR_OV_CONFIG_NAME
        if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists() or not self.ov_config_path.exists():
            self.converted_to_ov = True
            print(f"### ov model files not found: "
                  f"ov_encoder_path={self.ov_encoder_path}, "
                  f"ov_decoder_path={self.ov_decoder_path}, "
                  f"ov_config_path={self.ov_config_path}")

    def load_ov_config(self):
        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]

            self.config = GlmAsrConfig.from_pretrained(self.ov_config_path.parent)
            self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent)
            
            if self.generation_config.pad_token_id is None :
                if isinstance(self.generation_config.eos_token_id, list) :
                    self.generation_config.pad_token_id = self.generation_config.eos_token_id[0]
                else :
                    self.generation_config.pad_token_id = self.generation_config.eos_token_id

        except Exception as e:
            print(f"### ov load {self.ov_config_path} failed, {e}")

    def load_ov_model(self):
        try :            
            if self.ov_core is None :
                self.ov_core = Core()
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
            audio_embeds = np.zeros((input_ids.shape[0], 1))
            audio_token_mask = np.array([False]).reshape((input_ids.shape[0],1,1))

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
        self._past_length += input_ids.shape[1]
        out = CausalLMOutputWithPast(logits=logits, past_key_values=((),))
        return out

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # print(f"### GlmAsrEncDecModel::prepare_inputs_for_generation kwargs keys={list(kwargs.keys())}")
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

class GlmAsrEncDecModel1(GlmAsrEncDecModel) :
    def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size):
        super().__init__(ov_core, model_path, enc_type, dec_type, cache_size)

    def init_model_path(self, ov_path):
        self.ov_audio_encoder_path = ov_path  / GLMASR_Audio_Encoder_MODEL_NAME
        self.ov_input_encoder_path = ov_path / GLMASR_Input_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path  / GLMASR_Decoder_MODEL_NAME
        self.ov_config_path = ov_path  / GLMASR_OV_CONFIG_NAME
        if not self.ov_audio_encoder_path.exists() or not self.ov_input_encoder_path.exists() or not self.ov_decoder_path.exists() or not self.ov_config_path.exists():
            self.converted_to_ov = True
            print(f"### ov model files not found: "
                  f"ov_encoder_path={self.ov_encoder_path}, "
                  f"ov_decoder_path={self.ov_decoder_path}, "
                  f"ov_config_path={self.ov_config_path}")

    def load_ov_model(self):
        try :            
            if self.ov_core is None :
                self.ov_core = Core()
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
        example_inputs = {"input_ids":input_ids}
        self.input_enc_request.start_async(example_inputs, share_inputs=True)
        if input_features is not None:
            example_inputs = {"input_features":input_features, "input_features_mask":input_features_mask}
            self.audio_enc_request.start_async(example_inputs, share_inputs=True)
            self.dec_request.reset_state()
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0
            self.audio_enc_request.wait()
            self.input_enc_request.wait()
            audio_embeds = torch.from_numpy(self.audio_enc_request.get_output_tensor(0).data)
            inputs_embeds = torch.from_numpy(self.input_enc_request.get_output_tensor(0).data)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(audio_token_mask, audio_embeds)
        else :
            self.input_enc_request.wait()
            inputs_embeds = self.input_enc_request.get_output_tensor(0).data


        example_inputs = {"inputs_embeds":inputs_embeds,
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

FUNASR_Audio_Encoder_MODEL_NAME = "funasr_audio_encoder.xml"
FUNASR_Audio_Encoder_CTC_MODEL_NAME = "funasr_audio_encoder_ctc.xml"
FUNASR_Input_Encoder_MODEL_NAME = "funasr_input_encoder.xml"
FUNASR_Decoder_MODEL_NAME = "funasr_llm_decoder.xml"
FUNASR_OV_CONFIG_NAME = "ov_config.yaml"
FUNASR_Frontend_CONFIG_NAME = "frontend_config.json"


def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    items = []
    try:
        import torchaudio.functional as TAF
        from itertools import groupby
        # The current version only supports batch_size==1.
        log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
        assert log_probs.shape[1] >= targets.shape[1]
        alignments, scores = TAF.forced_align(log_probs, targets, blank=blank)
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
    except Exception as e:
        print(f"### forced_align failed: {e}")
    return items


class FunAsrNanoEncDecModel(BaseEncDecGenModel) :
    def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size, for_dialect=True, disable_ctc=False):
        self.disable_ctc = disable_ctc
        self.load_ov_config_once = False
        self.for_dialect = for_dialect
        super().__init__(ov_core, model_path, enc_type, dec_type, cache_size)
        self.frontend = self.load_frontend_from_config()
        self.tokenizer = self.load_tokenizer()

    def init_model_path(self, ov_path):
        self.ov_config_path = ov_path  / FUNASR_OV_CONFIG_NAME
        self.load_ov_config()
        if self.using_ctc:
            self.ov_audio_path = ov_path  / FUNASR_Audio_Encoder_CTC_MODEL_NAME
        else :
            self.ov_audio_path = ov_path  / FUNASR_Audio_Encoder_MODEL_NAME
        self.ov_text_path = ov_path  / FUNASR_Input_Encoder_MODEL_NAME
        self.ov_decoder_path = ov_path  / FUNASR_Decoder_MODEL_NAME
        self.ov_frontend_path = ov_path  / FUNASR_Frontend_CONFIG_NAME
        if (not self.ov_audio_path.exists() or not self.ov_text_path.exists()
            or not self.ov_decoder_path.exists() or not self.ov_config_path.exists() ):
            self.converted_to_ov = True
            print(f"### ov model files not found: "
                  f"ov_audio_path={self.ov_audio_path}, "
                  f"ov_text_path={self.ov_text_path}, "
                  f"ov_decoder_path={self.ov_decoder_path}, "
                  f"ov_config_path={self.ov_config_path}")

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.ov_config_path.parent, fix_mistral_regex=True, trust_remote_code=True,)
        return tokenizer

    def load_ctc_tokenizer(self):
        from funasr.register import tables
        ctc_tokenizer_class = tables.tokenizer_classes.get(self.ctc_tokenizer_name)
        ctc_tokenizer = ctc_tokenizer_class(**self.ctc_tokenizer_conf)
        return ctc_tokenizer
        
    def load_frontend_from_config(self):
        from funasr.register import tables
        with open(self.ov_frontend_path, "r") as f:
            frontend_config = json.load(f)
        frontend_type = frontend_config.pop("frontend_type", "WavFrontend")
        frontend_class = tables.frontend_classes.get(frontend_type)
        frontend = frontend_class(**frontend_config)
        return frontend

    def load_ov_config(self):
        if self.load_ov_config_once:
            return
        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data.get("main_input_name", "input_ids")
                self.use_low_frame_rate = data.get("use_low_frame_rate", False)
                self.pad_token_id = data.get("pad_token_id", None)
                if self.disable_ctc:
                    self.using_ctc = False
                else :
                    self.using_ctc = data.get("using_ctc", False)
                if self.using_ctc:
                    print(f"Load CTC tokenizer from {self.ov_config_path.parent}")
                    self.blank_id = data.get("blank_id", False)
                    self.ctc_tokenizer_name = data.get("ctc_tokenizer", "SenseVoiceTokenizer")
                    self.ctc_tokenizer_conf = data.get("ctc_tokenizer_conf", "{}")
                    vocab_path = self.ctc_tokenizer_conf.get("vocab_path", None)
                    if vocab_path is not None:
                        self.ctc_tokenizer_conf["vocab_path"] = str(self.ov_config_path.parent / vocab_path)
                    
            self.config = AutoConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)

            if self.using_ctc:
                self.ctc_tokenizer = self.load_ctc_tokenizer()
            self.load_ov_config_once = True
        except Exception as e:
            print(f"### ov load {self.ov_config_path} failed, {e}")

    def load_ov_model(self):
        try :            
            if self.ov_core is None :
                self.ov_core = Core()
            cache_size_str = f"{self.cache_size}"
            self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
           
            device = "CPU"
            ov_config = {}
            ov_config['NUM_STREAMS'] = 1
            ov_config['PERF_COUNT'] = 'NO'
            ov_config['PERFORMANCE_HINT'] = 'LATENCY'
           
            ov_config['INFERENCE_PRECISION_HINT'] = self.dec_type
            model = self.ov_core.read_model(self.ov_decoder_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.dec_request = compiled_model.create_infer_request()

            ov_config['INFERENCE_PRECISION_HINT'] = self.enc_type
            model = self.ov_core.read_model(self.ov_text_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.text_request = compiled_model.create_infer_request()

            ov_config['SNIPPETS_MODE'] = 'DISABLE'
            model = self.ov_core.read_model(self.ov_audio_path)
            compiled_model = self.ov_core.compile_model(model, device, ov_config)
            self.audio_request = compiled_model.create_infer_request()

            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_audio_path} or {self.ov_text_path} "
                  f"or {self.ov_decoder_path} failed, {e}")

    def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
        if self.for_dialect :
            return f"语种方言识别："
        if len(hotwords) > 0:
            hotwords = ", ".join(hotwords)
            prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{hotwords}]\n"
        else:
            prompt = ""
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def generate_chatml(self, prompt: str, data: Union[str, torch.Tensor]):
        if isinstance(data, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>"},
                {"role": "assistant", "content": "null"},
            ]
        elif isinstance(data, torch.Tensor):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>",
                    "audio": data,
                },
                {"role": "assistant", "content": "null"},
            ]

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = True
        sys_prompt = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = (
                            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        )
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"
            if kwargs.get("prev_text", None) is not None:
                source_input += kwargs["prev_text"]

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(
                                sub_str, fs=frontend.fs, **kwargs
                            )
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            import traceback
                            print(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if self.use_low_frame_rate:
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                        else:
                            fake_token_len_i = speech_lengths[0].item()
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def preprocess(self, data_in, data_lengths=None, key: list = None, tokenizer=None, frontend=None, **kwargs,):
        if tokenizer is None :
            tokenizer = self.tokenizer
        if frontend is None :
            frontend = self.frontend
        meta_data = {}
        prompt = self.get_prompt(
            kwargs.get("hotwords", []), kwargs.get("language", None), kwargs.get("itn", True)
        )
        data_in = [self.generate_chatml(prompt, data) for data in data_in]

        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append("rand_key_" + "".join(random.choice(chars) for _ in range(13)))

        contents = self.data_template(data_in[0])
        outputs = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        return outputs, contents, key, meta_data
    
    def predict(self, outputs, key, **kwargs,):
        speech = outputs['speech']
        speech_lengths = outputs['speech_lengths'][:, 0]
        self.audio_request.start_async({"speech":speech, "speech_lengths":speech_lengths}, share_inputs=True)
        
        input_ids = outputs['source_ids']
        input_ids[input_ids < 0] = 0
        self.text_request.start_async({"input_ids":input_ids}, share_inputs=True)

        fake_token_len = outputs['fake_token_len']
        fbank_beg = outputs['fbank_beg']
        attention_mask = outputs["attention_mask"]

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0
        
        #first inference clear dec state
        self._past_length = 0
        self.dec_request.reset_state()
        self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)

        self.audio_request.wait()
        self.text_request.wait()
        
        adaptor_out = torch.from_numpy(self.audio_request.get_output_tensor(0).data)
        adaptor_out_lens = torch.from_numpy(self.audio_request.get_output_tensor(1).data)
        inputs_embeds = torch.from_numpy(self.text_request.get_output_tensor(0).data)

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
                        print(f"#### patch inputs_embeds, erro={e}")
                        speech_token_len = adaptor_out_lens[speech_idx].item()
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1

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
            ctc_logits = torch.from_numpy(self.audio_request.get_output_tensor(2).data)
            yseqs = torch.from_numpy(self.audio_request.get_output_tensor(3).data)
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

        return generated_ids, ctc_results

    def postprocess(self, generated_ids, ctc_results, contents, key, meta_data, tokenizer, **kwargs,):
        if self.tokenizer is not None :
            tokenizer = self.tokenizer

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

    def inference(self, data_in, data_lengths=None, key: list = None, tokenizer=None, frontend=None, **kwargs,):
        if tokenizer is None :
            tokenizer = self.tokenizer
        if frontend is None :
            frontend = self.frontend
        outputs, contents, key, meta_data = self.preprocess(data_in, data_lengths, key, tokenizer, frontend, **kwargs)
        generated_ids, ctc_results = self.predict(outputs, key, **kwargs)
        return self.postprocess(generated_ids, ctc_results, contents, key, meta_data, tokenizer, **kwargs)
 
    def forward(self, inputs_embeds, attention_mask, past_key_values=None, **kwargs):
        self.dec_request.start_async({"inputs_embeds":inputs_embeds,
                                      "attention_mask":attention_mask,
                                      "beam_idx": self.next_beam_idx}, share_inputs=True)
        self.dec_request.wait()
        logits = torch.from_numpy(self.dec_request.get_tensor("logits").data)
        past_key_values = ((),)
        self._past_length += inputs_embeds.shape[1]
        outputs = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds, attention_mask, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(input_ids=input_ids,
                                                             inputs_embeds=inputs_embeds,
                                                             attention_mask=attention_mask,
                                                             **kwargs)
        if model_inputs["inputs_embeds"] is None:
            self.text_request.start_async({"input_ids":model_inputs["input_ids"]}, share_inputs=True)
            self.text_request.wait()
            model_inputs["inputs_embeds"] = torch.from_numpy(self.text_request.get_output_tensor(0).data)
        return model_inputs

class UnimernetEncoderModel(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=True, 
                    means=[0.7931, 0.7931, 0.7931], 
                    scales=[0.1738, 0.1738, 0.1738]) :
        # ppp = PrePostProcessor(self.model)
        # ppp.input(self.input_name).tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_color_format(ColorFormat.BGR) \
        #     .set_layout(Layout('NHWC'))


        # ppp.input(self.input_name).model() \
        #     .set_layout(Layout('NCHW'))

        # ppp.input(self.input_name).preprocess() \
        #     .convert_element_type(Type.f32) \
        #     .mean([x*255.0 for x in means])  \
        #     .scale([x*255.0 for x in scales]) 


        # self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, pixel_values):
        output = self.request.infer(pixel_values)
        return output
        
class UnimernetDecoderModel(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=True, means=[0.7931, 0.7931, 0.7931], 
                    scales=[0.1738, 0.1738, 0.1738], shape=[1, 1280, 1280, 3]) :
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i][0])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i][0]))
        return res
    
    def clear_requests(self) :
        if self.request:
            self.request.reset_state()
        if self.infer_queue:
            self.infer_queue.reset_state()

class PaddleTextClsProcessor(OV_Operator):
    def __call__(self, args):
        return self.request.infer(args)
    
class RapidTableProcesser(OV_Operator):
    def __call__(self, args):
        return self.request.infer(args)

class YoloProcessor(OV_Operator):
    def __call__(self, args):
        return self.request.infer(args)

class YoloV8OVProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False, shape=None,
                    means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225]) :
        ppp = PrePostProcessor(self.model)
        # print(f"self.input_names={self.input_names}")
        ppp.input(self.input_names[0]).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_names[0]).model() \
            .set_layout(Layout('NCHW'))

        # .resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, shape[1], shape[2]) \
        ppp.input(self.input_names[0]).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res   
    
class ClipSegProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    means=[0.485, 0.456, 0.406],
                    scales=[0.229, 0.224, 0.225],
                    shape=[1, 352, 352, 3]) :
        ppp = PrePostProcessor(self.model)
        # print(f"self.input_names={self.input_names}")
        ppp.input(self.input_names[0]).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_names[0]).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_names[0]).preprocess() \
            .resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, shape[1], shape[2]) \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 

        # ppp.input(self.input_names[1]).tensor() \
        #     .set_shape([shape[0], -1])    

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
class EfficientMMOENetProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    means=[0.485, 0.456, 0.406],
                    scales=[0.229, 0.224, 0.225],
                    shape=[1, 224, 224, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_names[0]).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_names[0]).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_names[0]).preprocess() \
            .resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, shape[1], shape[2]) \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 

        ppp.input(self.input_names[1]).tensor() \
            .set_shape([shape[0], -1])    

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class AudioProjProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    means=[0.485, 0.456, 0.406],
                    scales=[0.229, 0.224, 0.225],
                    shape=[1, 1280, 1280, 3]) :
        # ppp = PrePostProcessor(self.model)
        # ppp.input(self.input_name).tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_shape(shape) \
        #     .set_color_format(ColorFormat.BGR) \
        #     .set_layout(Layout('NHWC'))
        #     # 


        # ppp.input(self.input_name).model() \
        #     .set_layout(Layout('NCHW'))

        # ppp.input(self.input_name).preprocess() \
        #     .convert_color(ColorFormat.RGB) \
        #     .convert_element_type(Type.f32) \
        #     .mean([x*255.0 for x in means])  \
        #     .scale([x*255.0 for x in scales]) 


        # self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class FaceLocaterProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    means=[0.485, 0.456, 0.406],
                    scales=[0.229, 0.224, 0.225],
                    shape=[1, 1280, 1280, 3]) :
        # ppp = PrePostProcessor(self.model)
        # ppp.input(self.input_name).tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_shape(shape) \
        #     .set_color_format(ColorFormat.BGR) \
        #     .set_layout(Layout('NHWC'))
        # ppp.input(self.input_name).model() \
        #     .set_layout(Layout('NCHW'))
        # ppp.input(self.input_name).preprocess() \
        #     .convert_color(ColorFormat.RGB) \
        #     .convert_element_type(Type.f32) \
        #     .mean([x*255.0 for x in means])  \
        #     .scale([x*255.0 for x in scales]) 
        # self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
class DenoiseUnetProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False, shape=None) :
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class ReferenceUnetProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    scale = 0.18215,
                    shape=[-1, 4, 48, 48]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_names[1]).tensor() \
            .set_shape(shape) \
            .set_layout(Layout('NCHW'))
        ppp.input(self.input_names[1]).model() \
            .set_layout(Layout('NCHW'))
        ppp.input(self.input_names[1]).preprocess() \
            .scale(1.0/scale) 
        # ppp.input(self.input_names[0]).tensor() \
        #     .set_shape([-1]) 
        self.model = ppp.build()
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class VaeEncProcessor(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    mean=127.5, scale= 127.5,
                    shape=[1, 384, 384, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR, shape[1], shape[2]) \
            .mean(127.5) \
            .scale(127.5)

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class VaeDecProcessor(OV_Operator):
    def setup_model(self, stream_num = 4, bf16=True, f16=False,
                    scale=0.18215, shape=[1, 384, 384, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .scale(scale)

        self.model = ppp.build()
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i][0])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i][0]))
        return res

    
    def clear_requests(self) :
        if self.request:
            self.request.reset_state()
        if self.infer_queue:
            self.infer_queue.reset_state()

class DonutEncProcessor(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False,
                    means=[0.485, 0.456, 0.406],
                    scales=[0.229, 0.224, 0.225],
                    shape=[1, 1280, 1280, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))
            # 


        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 


        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class DonutDecProcessor(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False) :       
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
    def clear_requests(self) :
        if self.request:
            self.request.reset_state()
        if self.infer_queue:
            self.infer_queue.reset_state()
        
class LayoutLMv3Processor(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False,
                    means=[0.5, 0.5, 0.5],
                    scales=[0.5, 0.5, 0.5],
                    shape=[1, 224, 224, 3]) :
        # self.patch_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=torch.tensor((0.5, 0.5, 0.5)),
        #         std=torch.tensor((0.5, 0.5, 0.5)))
        # ])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # size = (image.shape[1], image.shape[0])
        # image = Image.fromarray(image)
        # image = image.resize((224, 224), Image.LANCZOS)
        # patch = self.patch_transform(image)
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 
            # .resize(ResizeAlgorithm.RESIZE_LINEAR) \


        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)
 
        res = []
        if self.postprocess is None:
            for j in range(len(self.res.results[0])):
                res_list = []
                for i in range(nsize) :
                    res_list.append(self.res.results[i][j][0])
                res.append(res_list)
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class RelationsProcessor(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False, shape=None) :
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize=len(input_tensors)
        if nsize>1 or self.request is None:
            for i, input_tensor in input_tensors:
                self.infer_queue.start_async(input_tensor, userdata=i, share_inputs=True)
            self.infer_queue.wait_all()
        else :
            self.res.sync_clean()
            for i, input_tensor in input_tensors:
                result = self.request.infer(input_tensor)
                self.res.sync_parser(result, i)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
   
class Fingerprint(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False, shape=None) :       
        super().setup_model(stream_num, bf16, f16, shape)
        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensor) :
        if self.infer_queue:
            self.infer_queue.start_async({0: input_tensor}, userdata=0, share_inputs=True)
        self.infer_queue.wait_all()
       
        if self.postprocess is None:
               return self.res.results
        else :
           return self.postprocess(self.res.results)
        return res
    
class CTCSimpleOCR(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False, shape_static=None, shape_dynamic=None) :
        scale = [127.5]
        if shape_static is not None and shape_dynamic is not None:
            self.model_dynamic = self.model.clone()
            ppp_dyn = PrePostProcessor(self.model_dynamic)
            ppp_dyn.input(self.input_name).tensor() \
                    .set_element_type(Type.u8) \
                    .set_shape(shape_dynamic) \
                    .set_layout(Layout('NHWC')) 
            ppp_dyn.input(self.input_name).model().set_layout(Layout('NCHW'))
            ppp_dyn.input(self.input_name).preprocess() \
                .convert_element_type(Type.f32) \
                .mean(scale) \
                .scale(scale)
            self.model_dynamic = ppp_dyn.build()
            shape = shape_static       
        else :
            if shape_static is not None :
                shape = shape_static
            elif shape_dynamic is not None :
                shape = shape_dynamic
            else :
                shape = None
        ppp = PrePostProcessor(self.model)
        if shape is None:
            ppp.input(self.input_name).tensor() \
                .set_element_type(Type.u8) \
                .set_layout(Layout('NHWC')) 
        else :
            ppp.input(self.input_name).tensor() \
                .set_element_type(Type.u8) \
                .set_shape(shape) \
                .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()

        super().setup_model(stream_num, bf16, f16, None)

        self.ocr_res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.ocr_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        if self.request and len(norm_img_batch_list)==1:
            for i, input_tensor in enumerate(norm_img_batch_list):
                result = self.request.infer(input_tensor)
                self.ocr_res.sync_parser(result, 0)
            return self.ocr_res.results 
        
        nsize=len(norm_img_batch_list)
        dyanmic_list = []
        for i, input_tensor in enumerate(norm_img_batch_list):
            if self.model_dynamic is not None and input_tensor.shape[2] ==320:
                self.infer_queue.start_async({0: input_tensor}, userdata=i)
            else :
                dyanmic_list.append((i,input_tensor))
        self.infer_queue.wait_all()
        for i, input_tensor in dyanmic_list:
            result = self.request.infer(input_tensor)
            self.ocr_res.sync_parser(result, i)
            
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.ocr_res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.ocr_res.results[i]))
        return res
    
    # def __call__(self, static_list, dyanmic_list) :
    #     nsize=len(static_list)
    #     for i, input_tensor in enumerate(static_list):
    #         self.infer_queue.start_async({0: input_tensor}, userdata=i)
            
    #     self.infer_queue.wait_all()
        
    #     for i, input_tensor in enumerate(dyanmic_list):
    #         result = self.request.infer(input_tensor)
    #         self.ocr_res.sync_parser(result, nsize+i)
    #     nsize += len(dyanmic_list)
    #     res = []
    #     if self.postprocess is None:
    #         for i in range(nsize) :
    #             res.append(self.ocr_res.results[i])
    #     else :
    #         for i in range(nsize) :
    #             res.append(self.postprocess(self.ocr_res.results[i]))    
    #     return res

class SqlBertProcessor(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False, shape=None) :
        super().setup_model(stream_num, bf16, f16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize=len(input_tensors)
        if self.request :
            self.res.sync_clean()
            for i, input_tensor in enumerate(input_tensors):
                result = self.request.infer(input_tensor)
                self.res.sync_parser(result, i)
        else :
            for i, input_tensor in enumerate(input_tensors):
                self.infer_queue.start_async(input_tensor, userdata=i)
            self.infer_queue.wait_all()
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
    def __async_call_(self, input_tensors):
        res = []
        for input_tensor in input_tensors:
            idle_id = self.infer_queue.get_idle_request_id()
            res.append(self.res.results[idle_id])
            self.infer_queue.start_async(input_tensor, userdata=idle_id)
        return res

class ObjDetector(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False,
                    shape=[1, 3, 512, 512]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NCHW')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))


        # mean = [123.675, 116.28, 103.53]
        # scale = [58.395, 57.12, 57.375]
        # ppp.input(self.input_name).preprocess() \
        #     .convert_element_type(Type.f32) \
        #     .mean(mean)  \
        #     .scale(scale) 

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)
        self.det_res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.det_res.completion_callback)


    def __call__(self, images) :   
        nsize=len(images)
        if self.request :
            self.det_res.sync_clean()
            for i, image in enumerate(images):
                result = self.request.infer({0: image})
                self.det_res.sync_parser(result, i)
        else :
            for i, image in enumerate(images):
                self.infer_queue.start_async({0: image}, userdata=i)
            self.infer_queue.wait_all()
            
        if self.postprocess is None:
            return self.det_res.results[0][0]
        else :
            return self.postprocess(self.det_res.results[0][0])

class PaddleTextDetector(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False) :       
        super().setup_model(stream_num, bf16, f16, None)
        self.det_res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.det_res.completion_callback)

    def __call__(self, images) :   
        nsize=len(images)
        if self.request :
            self.det_res.sync_clean()
            for i, image in enumerate(images):
                result = self.request.infer({0: image})
                self.det_res.sync_parser(result, i)
        else :
            for i, image in enumerate(images):
                self.infer_queue.start_async({0: image}, userdata=i)
            self.infer_queue.wait_all()
            
        if self.postprocess is None:
            return self.det_res.results[0][0]
        else :
            return self.postprocess(self.det_res.results[0][0])

class TextDetector(OV_Operator):
    def setup_model(self, stream_num = 1, bf16=True, f16=False, shape=[1, -1,-1, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))


        mean = [123.675, 116.28, 103.53]
        scale = [58.395, 57.12, 57.375]
        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(mean)  \
            .scale(scale) 

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)
        self.det_res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.det_res.completion_callback)

    def __call__(self, images) :   
        nsize=len(images)
        if self.request :
            self.det_res.sync_clean()
            for i, image in enumerate(images):
                result = self.request.infer({0: image})
                self.det_res.sync_parser(result, i)
        else :
            for i, image in enumerate(images):
                self.infer_queue.start_async({0: image}, userdata=i)
            self.infer_queue.wait_all()
            
        if self.postprocess is None:
            return self.det_res.results[0][0]
        else :
            return self.postprocess(self.det_res.results[0][0])

class TextRecognizerOV(OV_Operator):
    def setup_model(self, stream_num = 2, bf16=True, f16=False, shape=[-1,32,-1,3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        scale = [127.5, 127.5, 127.5]
        
        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)

        self.ocr_res = OV_Result(self.outputs)
        self.infer_queue.set_callback(self.ocr_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        nsize=len(norm_img_batch_list)
        #for i in range(nsize-1, -1, -1):
        #    self.infer_queue.start_async({0: norm_img_batch_list[i]}, userdata=i)

        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)
            
        self.infer_queue.wait_all()

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.ocr_res.results[0][i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.ocr_res.results[0][i]))
        return res

class TextClassfier(OV_Operator):
    def setup_model(self, stream_num=2, bf16=True, f16=False, shape=[-1,32,-1,3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        scale = [127.5, 127.5, 127.5]

        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, f16, None)
        self.cls_res = OV_Result(self.outputs)
        self.infer_queue.set_callback(self.cls_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)

        self.infer_queue.wait_all()

        res = []
        if self.postprocess is None:
            for i in range(len(norm_img_batch_list)) :
                res.append(self.cls_res.results[0][i])
        else :
            for i in range(len(norm_img_batch_list)) :
                res.append(self.postprocess(self.cls_res.results[0][i]))
        return res
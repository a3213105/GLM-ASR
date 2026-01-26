from array import array
from locale import ABDAY_1
import numpy as np
from datetime import datetime
from openvino import Core,Model, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
import os
import copy
from pathlib import Path
    
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

from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig
import torch

class GlmAsrEncDecModel(GenerationMixin) :
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
            self.load_ov_model()

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
        
    def load_ov_model(self):
        try :            
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]

            self.config = AutoConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            
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
            import yaml
            with open(self.ov_config_path, "r") as f:
                data = yaml.safe_load(f)
                self.main_input_name = data["main_input_name"]

            self.config = AutoConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            self.generation_config = GenerationConfig.from_pretrained(self.ov_config_path.parent, trust_remote_code=True,)
            
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
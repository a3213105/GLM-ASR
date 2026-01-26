from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
import torch
import time
import argparse
from pathlib import Path

from transformers.generation import GenerationMixin
from ov_model_helper import GlmAsrForOVConvertWrapper, GlmAsrForOVConvertWrapper1

import transformers
from packaging import version
transformers_ver = version.parse(transformers.__version__)


parser = argparse.ArgumentParser(description="Minimal ASR transcription demo.")
parser.add_argument("--checkpoint_dir", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512/")
parser.add_argument("--ov_mode_dir", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512-ov/")
parser.add_argument("--llm_output_dir", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512-llm/")
parser.add_argument("--checkpoint_dir4", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512-llm/")
args = parser.parse_args()

def save_llama_to_transformer4(model, args) :
    if args.llm_output_dir is None:
        args.llm_output_dir = "/tmp/llm"
    model.language_model.save_pretrained(args.llm_output_dir)
    print(f"language_model saved to {args.llm_output_dir}, convert later with transformers 4.x")
    
def convert_llama_to_ov(args) :
    processor = None
    llm_model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir4, device_map="cpu", trust_remote_code=True)
    llm_model.config._attn_implementation = "eager"
    llm_model = llm_model.float()
    llm_model.eval()
    class GLMASRWrapper(torch.nn.Module) :
        def __init__(self, llm_model):
            super().__init__()
            self.language_model = llm_model.eval()
            self.config = llm_model.config
            self.generation_config = llm_model.generation_config
            self.main_input_name = llm_model.main_input_name

    model = GLMASRWrapper(llm_model)

    print("Preparing dummy inputs for conversion...")
    seq_len = 90
    input_ids=torch.randint(0, 1000, (1, 1), dtype=torch.long)
    audio_embeds = torch.zeros((input_ids.shape[0], 1))
    audio_token_mask = torch.tensor([False]).reshape((input_ids.shape[0],1,1))
    attention_mask = torch.ones((input_ids.shape[0], seq_len+1), dtype=torch.long)
    position_ids=torch.tensor([[seq_len]], dtype=torch.long)
    cache_position=torch.tensor([seq_len], dtype=torch.long)
    inputs_embeds=torch.randn((input_ids.shape[0], 1, 2048), dtype=torch.float)
    past_key_values = []
    for _ in range(28):
        key = torch.randn((1, 4, seq_len, 128), dtype=torch.float)
        value = torch.randn((1, 4, seq_len, 128), dtype=torch.float)
        past_key_values.append((key, value))

    dec_wrapper = GlmAsrForOVConvertWrapper(model, processor, args.ov_mode_dir + "/ov_model0")
    dec_wrapper.convert_decoder_to_ov(input_ids=input_ids, audio_embeds=audio_embeds,
                                      audio_token_mask=audio_token_mask, attention_mask=attention_mask,
                                      position_ids=position_ids, cache_position=cache_position,
                                      past_key_values=past_key_values)

    dec_wrapper1 = GlmAsrForOVConvertWrapper1(model, processor, args.ov_mode_dir + "/ov_model1")
    dec_wrapper1.convert_decoder_to_ov(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       position_ids=position_ids, cache_position=cache_position,
                                      past_key_values=past_key_values)
    
def convert_other_to_ov(args) :
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir, device_map="cpu")

    model = AutoModel.from_pretrained(args.checkpoint_dir, dtype=torch.float, device_map="cpu")
    model = model.float()
    model.eval()

    #input_features=torch.Size([1, 128, 3000]), input_features_mask=torch.Size([1, 3000])
    seq_len = 90
    input_ids=torch.randint(0, 1000, (1, 90), dtype=torch.long)
    input_features = torch.randn((input_ids.shape[0], 128, 3000), dtype=torch.float)
    input_features_mask = torch.randint(0, 1, (1, 3000), dtype=torch.long)

    dec_wrapper = GlmAsrForOVConvertWrapper(model, processor, args.ov_mode_dir + "/ov_model0")
    dec_wrapper.convert_encoder_to_ov(input_ids, input_features, input_features_mask)

    dec_wrapper1 = GlmAsrForOVConvertWrapper1(model, processor, args.ov_mode_dir + "/ov_model1")
    dec_wrapper1.convert_inputs_emb_to_ov(input_ids)
    dec_wrapper1.convert_audio_emb_to_ov(input_features, input_features_mask)
    
    save_llama_to_transformer4(model, args)

if __name__ == "__main__":
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.ov_model_dir = Path(args.ov_mode_dir)
    if transformers_ver.major >= 5:
        convert_other_to_ov(args)
    else :
        convert_llama_to_ov(args)
    
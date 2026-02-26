from transformers import AutoModel, AutoProcessor
import torch
import time
import argparse
from pathlib import Path

from transformers.generation import GenerationMixin
# from transformers.modeling_utils import PreTrainedModel
from ov_model_helper import GlmAsrForOVConvertWrapper, GlmAsrForOVConvertWrapper1
from ov_operator_async import GlmAsrEncDecModel, GlmAsrEncDecModel1
 

parser = argparse.ArgumentParser(description="Minimal ASR transcription demo.")
parser.add_argument("--checkpoint_dir", "-c", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512/")
parser.add_argument("--ov_model_dir", "-o", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512-ov1/")
parser.add_argument("--audio", "-a", type=str, default="examples/example_zh.wav",
                    help="Path to audio file.")
parser.add_argument("--tokenizer_path", "-t", type=str, default=None,
                    help="Tokenizer directory (defaults to checkpoint dir when omitted).",)
parser.add_argument("--max_new_tokens", "-m", type=int, default=128)
parser.add_argument("--loop", "-l", type=int, default=10)
args = parser.parse_args()

try :
    model_bf16 = AutoModel.from_pretrained(args.checkpoint_dir, dtype=torch.bfloat16, device_map="cpu")
    model_bf16.eval()
except:
    model_bf16 = None
try:
    model_f32 = AutoModel.from_pretrained(args.checkpoint_dir, dtype=torch.float, device_map="cpu")
    model_f32 = model_f32.float()
    model_f32.eval()
except:
    model_f32 = None

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput

processor = AutoProcessor.from_pretrained(args.checkpoint_dir, device_map="cpu", trust_remote_code=True)
    
inputs_f32 = processor.apply_transcription_request(args.audio, return_tensors="pt")
inputs_bf16 = inputs_f32.copy()
inputs_bf16 = inputs_bf16.to("cpu", dtype=torch.bfloat16)

convert_model = True

if convert_model:
    ov_model = GlmAsrForOVConvertWrapper(model_f32, processor, args.ov_model_dir+"/ov_model0")
    ov_model1 = GlmAsrForOVConvertWrapper1(model_f32, processor, args.ov_model_dir+"ov_model1/")
else :
    ov_model = GlmAsrEncDecModel(ov_core=None, model_path=args.ov_model_dir+"/ov_model0", enc_type='bf16', dec_type='bf16', cache_size=1000)
    ov_model1 = GlmAsrEncDecModel1(ov_core=None, model_path=args.ov_model_dir+"ov_model1/", enc_type='f16', dec_type='bf16', cache_size=1000)

torch_outputs=[]
torch_outputs1=[]
print(f"#############################################")
start_time = time.perf_counter()
with torch.no_grad():
    if model_bf16 :
        torch_outputs = model_bf16.generate(**inputs_bf16, max_new_tokens=args.max_new_tokens, do_sample=False)
torch_warmup_time = time.perf_counter() - start_time
print(f"#############################################")
start_time = time.perf_counter()
with torch.no_grad():
    if model_f32 :
        torch_outputs1 = model_f32.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
torch_warmup_time1 = time.perf_counter() - start_time
print(f"#############################################")
start_time = time.perf_counter()
with torch.no_grad():
    ov_outputs = ov_model.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
ov_warmup_time = time.perf_counter() - start_time
print(f"#############################################")
start_time = time.perf_counter()
with torch.no_grad():
    ov_outputs1 = ov_model1.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
ov_warmup_time1 = time.perf_counter() - start_time
print(f"#############################################")

if torch.equal(torch_outputs, ov_outputs) == False:
    print("警告: Torch 和 OpenVINO 生成的结果不相等！")
    print("Torch 输出:", torch_outputs)
    print("OpenVINO 输出:", ov_outputs)

torch_gen_tokens = torch_outputs[:, inputs_bf16.input_ids.shape[1]:]
torch_decode_text=processor.batch_decode(torch_gen_tokens, skip_special_tokens=True)
torch_tokens_count = torch_gen_tokens.shape[1]

torch_gen_tokens1 = torch_outputs1[:, inputs_f32.input_ids.shape[1]:]
torch_decode_text1=processor.batch_decode(torch_gen_tokens1, skip_special_tokens=True)
torch_tokens_count1 = torch_gen_tokens1.shape[1]

ov_gen_tokens = ov_outputs[:, inputs_f32.input_ids.shape[1]:]
ov_decode_text=processor.batch_decode(ov_gen_tokens, skip_special_tokens=True)
ov_tokens_count = ov_gen_tokens.shape[1]

ov_gen_tokens1 = ov_outputs1[:, inputs_f32.input_ids.shape[1]:]
ov_decode_text1=processor.batch_decode(ov_gen_tokens1, skip_special_tokens=True)
ov_tokens_count1 = ov_gen_tokens1.shape[1]

print(f"Torch_F32  生成文本: {torch_decode_text1[0]}")
print(f"Torch_F32  生成token数: {torch_tokens_count1} 个")
print(f"Torch_BF16 生成文本: {torch_decode_text[0]}")
print(f"Torch_BF16 生成token数: {torch_tokens_count} 个")
print(f"OpenVINO   生成文本: {ov_decode_text[0]}")
print(f"OpenVINO   生成token数: {ov_tokens_count} 个")
print(f"OpenVINO1  生成文本: {ov_decode_text1[0]}")
print(f"OpenVINO1  生成token数: {ov_tokens_count1} 个")

print(f"\n开始循环评估（共执行 {args.loop} 次）...")
torch_end_time=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    with torch.no_grad():
        model_bf16.generate(**inputs_bf16, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch_end_time += (time.perf_counter() - start_time)

torch_end_time1=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    with torch.no_grad():
        model_f32.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch_end_time1 += (time.perf_counter() - start_time)


ov_end_time=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    with torch.no_grad():
        ov_model.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
    ov_end_time += (time.perf_counter() - start_time)

ov_end_time1=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    with torch.no_grad():
        ov_model1.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
    ov_end_time1 += (time.perf_counter() - start_time)

torch_avg_time = torch_end_time / args.loop
torch_avg_time1 = torch_end_time1 / args.loop
ov_avg_time = ov_end_time / args.loop
ov_avg_time1 = ov_end_time1 / args.loop

print("-" * 30)
# print("数据类型:", torch.bfloat16)
print(f"测试次数: {args.loop}")
print(f"Torch_F32 首次推理 (Warm-up): {torch_warmup_time1:.3f} 秒")
print(f"Torch_F32 后续平均耗时: {torch_avg_time1:.3f} 秒")
print(f"Torch_F32 生成速度: {torch_tokens_count1 / torch_avg_time1:.2f} tokens/s")
print(f"Torch_BF16 首次推理 (Warm-up): {torch_warmup_time:.3f} 秒")
print(f"Torch_BF16 后续平均耗时: {torch_avg_time:.3f} 秒")
print(f"Torch_BF16 生成速度: {torch_tokens_count / torch_avg_time:.2f} tokens/s")
print(f"OpenVINO 首次推理 (Warm-up): {ov_warmup_time:.3f} 秒")
print(f"OpenVINO 后续平均耗时: {ov_avg_time:.3f} 秒")
print(f"OpenVINO 生成速度: {ov_tokens_count / ov_avg_time:.2f} tokens/s")
print(f"OpenVINO 首次推理 (Warm-up): {ov_warmup_time1:.3f} 秒")
print(f"OpenVINO 后续平均耗时: {ov_avg_time1:.3f} 秒")
print(f"OpenVINO 生成速度: {ov_tokens_count1 / ov_avg_time1:.2f} tokens/s")
print("-" * 30)

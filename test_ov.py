import time
import argparse
from pathlib import Path
# from transformers.audio_utils import AudioInput, make_list_of_audio
# from transformers.feature_extraction_utils import BatchFeature
# from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
# from transformers.tokenization_utils_base import TextInput
from transformers import AutoProcessor


from ov_operator_async import GlmAsrEncDecModel, GlmAsrEncDecModel1
 

parser = argparse.ArgumentParser(description="Minimal ASR transcription demo.")
parser.add_argument("--ov_model_dir", "-o", type=str, default=f"{Path(__file__).parent}/../GLM-ASR-Nano-2512-ov/")
parser.add_argument("--audio", "-a", type=str, default="examples/example_zh.wav",
                    help="Path to audio file.")
parser.add_argument("--max_new_tokens", "-m", type=int, default=128)
parser.add_argument("--loop", "-l", type=int, default=10)
args = parser.parse_args()


processor_path = args.ov_model_dir+"/ov_model0"
processor = AutoProcessor.from_pretrained(processor_path, device_map="cpu")

if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
    print(f"tokenizer vocab size: {processor.tokenizer.vocab_size}")
else:
    print(f"processor don't have tokenizer")
    from ov_operator_async import GlmAsrProcessor
    processor = GlmAsrProcessor(feature_extractor=processor, model_path=processor_path+"/v4")
    
inputs_f32 = processor.apply_transcription_request(args.audio, return_tensors="pt")

ov_model = GlmAsrEncDecModel(ov_core=None, model_path=args.ov_model_dir+"/ov_model0", enc_type='bf16', dec_type='bf16', cache_size=1000)

ov_model1 = GlmAsrEncDecModel1(ov_core=None, model_path=args.ov_model_dir+"ov_model1/", enc_type='f16', dec_type='bf16', cache_size=1000)


start_time = time.perf_counter()
ov_outputs = ov_model.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
ov_warmup_time = time.perf_counter() - start_time
print(f"#############################################")
start_time = time.perf_counter()
ov_outputs1 = ov_model1.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
ov_warmup_time1 = time.perf_counter() - start_time
print(f"#############################################")

ov_gen_tokens = ov_outputs[:, inputs_f32.input_ids.shape[1]:]
ov_decode_text=processor.batch_decode(ov_gen_tokens, skip_special_tokens=True)
ov_tokens_count = ov_gen_tokens.shape[1]

ov_gen_tokens1 = ov_outputs1[:, inputs_f32.input_ids.shape[1]:]
ov_decode_text1=processor.batch_decode(ov_gen_tokens1, skip_special_tokens=True)
ov_tokens_count1 = ov_gen_tokens1.shape[1]

print(f"OpenVINO   生成文本: {ov_decode_text[0]}")
print(f"OpenVINO   生成token数: {ov_tokens_count} 个")
print(f"OpenVINO1  生成文本: {ov_decode_text1[0]}")
print(f"OpenVINO1  生成token数: {ov_tokens_count1} 个")

print(f"\n开始循环评估（共执行 {args.loop} 次）...")
ov_end_time=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    ov_model.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
    ov_end_time += (time.perf_counter() - start_time)

ov_end_time1=0.0
for i in range(args.loop):
    start_time = time.perf_counter()
    ov_model1.generate(**inputs_f32, max_new_tokens=args.max_new_tokens, do_sample=False)
    ov_end_time1 += (time.perf_counter() - start_time)

ov_avg_time = ov_end_time / args.loop
ov_avg_time1 = ov_end_time1 / args.loop

print("-" * 30)
# print("数据类型:", torch.bfloat16)
print(f"测试次数: {args.loop}")
print(f"OpenVINO 首次推理 (Warm-up): {ov_warmup_time:.3f} 秒")
print(f"OpenVINO 后续平均耗时: {ov_avg_time:.3f} 秒")
print(f"OpenVINO 生成速度: {ov_tokens_count / ov_avg_time:.2f} tokens/s")
print(f"OpenVINO 首次推理 (Warm-up): {ov_warmup_time1:.3f} 秒")
print(f"OpenVINO 后续平均耗时: {ov_avg_time1:.3f} 秒")
print(f"OpenVINO 生成速度: {ov_tokens_count1 / ov_avg_time1:.2f} tokens/s")
print("-" * 30)

1.  Convert Audio/Inputs embedding part  
- install Transformers==5.0.0.dev  
```bash
python convert_glmasr_decoder.py --checkpoint_dir /PATH/TO/GLMASR/MODEL/DIR --ov_mode_dir /PATH/TO/OUTPUT/OV_MODEL_DIR --llm_tmp_dir /PATH/TO/TMP/LLM/MODEL/DIR
```
after this step, GlmASR encoder embedding parts will be converted to OpenVINO and stored in `ov_mode_dir` and Language model of GlmASR will be stored in `llm_tmp_dir`  
- install Transformers==4.49.0  
```bash
python convert_glmasr_decoder.py --ov_mode_dir /PATH/TO/OUTPUT/OV_MODEL_DIR --llm_tmp_dir /PATH/TO/TMP/LLM/MODEL/DIR
```
other decoding model with KV-cache will be converted and stored in `ov_mode_dir`

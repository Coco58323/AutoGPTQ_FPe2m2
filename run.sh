# MODEL=/root/exp/AutoGPTQ/Qwen2.5-7B-FPe2m2_RQ_2048sample_fp16-FP8-Dynamic
MODEL=/root/data/model/Qwen2.5/Qwen2.5-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template
lm_eval --model hf \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks mmlu,gsm8k --batch_size 4 --apply_chat_template

#   --model_args pretrained=$MODEL,add_bos_token=True,gpu_memory_utilization=0.2 \

# fp5 weight, fp8 inference
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.4411|±  |0.0145|
# |             |       |none  |     0|acc_norm|↑  |0.4096|±  |0.0144|
# |arc_easy     |      1|none  |     0|acc     |↑  |0.6932|±  |0.0095|
# |             |       |none  |     0|acc_norm|↑  |0.5000|±  |0.0103|

# fp5 weight, fp16 inference
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.4514|±  |0.0145|
# |             |       |none  |     0|acc_norm|↑  |0.4155|±  |0.0144|
# |arc_easy     |      1|none  |     0|acc     |↑  |0.6919|±  |0.0095|
# |             |       |none  |     0|acc_norm|↑  |0.5008|±  |0.0103|

# fp16 weight, fp16 inference
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.4471|±  |0.0145|
# |             |       |none  |     0|acc_norm|↑  |0.4113|±  |0.0144|
# |arc_easy     |      1|none  |     0|acc     |↑  |0.6982|±  |0.0094|
# |             |       |none  |     0|acc_norm|↑  |0.4857|±  |0.0103|
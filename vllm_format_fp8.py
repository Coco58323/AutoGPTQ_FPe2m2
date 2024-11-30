# dependencies
# pip install llmcompressor
from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer

# path to Quant
MODEL_ID = "/root/exp/AutoGPTQ/Qwen2.5-0.5B-FPe2m2_RQ_2048sample_fp16"
# MODEL_ID = "/root/data/model/meta-llama/LLaMA-3-8B"
# model = SparseAutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
# oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID + "-FP8-Dynamic-w5"
# model.save_pretrained(SAVE_DIR)
# tokenizer.save_pretrained(SAVE_DIR)
# %%
from vllm import LLM
model = LLM(SAVE_DIR)
print(model.generate("Hello my name is"))
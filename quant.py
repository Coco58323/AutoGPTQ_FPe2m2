import json
import random
import time
from argparse import ArgumentParser

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TextGenerationPipeline,AutoModelForCausalLM

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset

def get_ours(nsamples, data_path, seqlen, model, hf_token, eval_mode=False):
    f = open(data_path, 'r', encoding='utf8')
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    import json
    contents = []
    gsm8k_contents = []
    ceval_contents = []
    chinese_exam = []
    loginqa = []
    logiqa = []
    cn_code = []
    en_instruct_f = []
    en_qa = []
    en_mbpp = []
    en_math = []
    mmlu_contents = []
    humaneval_contents = []
    for l in f:
        data = json.loads(l)
        contents.append(data)
        continue
        if data['source'] == "en:mmlu":
            mmlu_contents.append(data)
        elif data['source'] == "cn:ceval":
            ceval_contents.append(data)
        elif data['source'] == "en:math.gsm8k":
            gsm8k_contents.append(data)
        elif data['source'] == "en:code.conala":
            humaneval_contents.append(data)
        elif data['source'] == "cn:reasoning.chinese_exam":
            chinese_exam.append(data)
        elif data['source'] == "cn:reasoning.logiqa2.0":
            loginqa.append(data)
        elif data['source'] == "cn:reasoning.logiqa":
            logiqa.append(data)
        elif data['source'] == "en:daring_anteater.instruction_following":
            en_instruct_f.append(data)
        elif data['source'] == "en:reasoning.logiqa":
            en_qa.append(data)
        elif data['source'] == "en:code.mbpp_derive":
            en_mbpp.append(data)
        elif data['source'] == "cn:code.sft-complex":
            cn_code.append(data)
        elif data['source'] == "en:math.qwenmath_sft":
            en_math.append(data)
        else:
            contents.append(data)
    import random
    random.shuffle(contents)
    # contents = en_math[:300] + en_mbpp[:500] + mmlu_contents[:500] + cn_code[:500] + ceval_contents[:500] + gsm8k_contents[:500] + humaneval_contents[:500] + chinese_exam[:800] + loginqa[:500] + logiqa[:500] + en_instruct_f[:300] + en_qa[:500] + contents[:4120]
    # contents = en_math + en_mbpp + mmlu_contents + cn_code + ceval_contents + gsm8k_contents + humaneval_contents + chinese_exam + loginqa + logiqa + en_instruct_f + en_qa + contents

    from tqdm import tqdm
    new_content = []
    for c in tqdm(contents):
        msgs = c['messages']
        item = ''
        for msg in msgs:
            role, content = msg['role'], msg['content'].rstrip()
            item += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        new_content.append(item)
    import random
    random.shuffle(new_content)
    examples = []
    for c in new_content[:nsamples]:
        examples.append(tokenizer(c,return_tensors='pt',max_length=seqlen,truncation=True,padding='max_length'))

    new_examples = []
    for e in examples:
        ids = e['input_ids'][:seqlen]
        new_examples.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})

    del contents
    del examples
    torch.cuda.empty_cache()
    return new_examples


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, default="/root/data/model/Qwen2.5/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--quantized_model_dir", type=str, default="/root/exp/AutoGPTQ/Qwen2.5-0.5B-FPe2m2_RQ_ct")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--calib_data", type=str, default="/root/exp/moe-lora/data/v10.15.35_beta.jsonl")
    parser.add_argument(
        "--group_size",
        type=int,
        default=-1,
        help="group size, -1 means no grouping or full rank",
    )
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )
    parser.add_argument(
        "--save_and_reload",
        action="store_false",
        help="whether save quantized model to disk and reload back",
    )
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="max memory used to load model per gpu",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=int,
        default=None,
        help="max memory used to offload model to cpu",
    )
    parser.add_argument(
        "--quant_batch_size",
        type=int,
        default=1,
        help="examples batch size for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="whether to trust remote code when loading model",
    )
    parser.add_argument(
        "--ready",
        action="store_true",
        help="whether to quantize model",
    )
    parser.add_argument(
        "--fpe2m2_checkpoint_format",
        action="store_true",
        help="whether to evaluate"
    )
    args = parser.parse_args()

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.fpe2m2_checkpoint_format:
        checkpoint_format = 'fpe2m2'
        quantized_model_dir = args.quantized_model_dir + "_fpe2m2"
    else:
        checkpoint_format = 'fp16'
        quantized_model_dir = args.quantized_model_dir + "_fp16"
    quant_config = BaseQuantizeConfig(bits=5,exp=2, group_size=args.group_size, desc_act=args.desc_act,checkpoint_format=checkpoint_format, mixed_precision=False)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=quant_config,
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    if not args.ready:
        examples = get_ours(args.num_samples, args.calib_data, 2048, args.pretrained_model_dir, tokenizer, eval_mode=True)
        examples_for_quant = [
            {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples
        ]
        start = time.time()
        model.quantize(
            examples_for_quant,
            batch_size=args.quant_batch_size,
            fpe2m2_checkpoint_format=args.fpe2m2_checkpoint_format,
            cache_examples_on_gpu=False,
        )
        end = time.time()
        print(f"quantization took: {end - start: .4f}s")

        # if args.save_and_reload:
        import os
        import shutil
        os.makedirs(quantized_model_dir, exist_ok=True)
        for file in os.listdir(args.pretrained_model_dir):
            if file.endswith(".txt") or file.endswith(".json"):
                shutil.copy(os.path.join(args.pretrained_model_dir, file), quantized_model_dir)
        if args.fpe2m2_checkpoint_format:
            model.save_quantized(quantized_model_dir)
        else:
            model.save_pretrained(quantized_model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("load model")
    if args.fpe2m2_checkpoint_format:
        quant_config = BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act,checkpoint_format='fpe2m2')
        model = AutoGPTQForCausalLM.from_quantized(
            quantized_model_dir,
            quantize_config=quant_config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_dir,
            trust_remote_code=args.trust_remote_code,
        )
    text = "Instruction:\nWrite a summary.\nInput:We consider the problem of model compression for deep neural networks (DNNs) in the challenging one-shot/post-training setting, in which we are given an accurate trained model, and must compress it without any retraining, based only on a small amount of calibration input data. This problem has become popular in view of the emerging software and hardware support for executing models compressed via pruning and/or quantization with speedup, and well-performing solutions have been proposed independently for both compression approaches. In this paper, we introduce a new compression framework which covers both weight pruning and quantization in a unified setting, is time- and space-efficient, and considerably improves upon the practical performance of existing post-training methods. At the technical level, our approach is based on an exact and efficient realization of the classical Optimal Brain Surgeon (OBS) framework of [LeCun, Denker, and Solla, 1990] extended to also cover weight quantization at the scale of modern DNNs. From the practical perspective, our experimental results show that it can improve significantly upon the compression-accuracy trade-offs of existing post-training methods, and that it can enable the accurate compound application of both pruning and quantization in a post-training setting."
    inputs = tokenizer(text, return_tensors="pt",max_length=128,padding=True,truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=2048, num_beams=1, no_repeat_ngram_size=2, early_stopping=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()

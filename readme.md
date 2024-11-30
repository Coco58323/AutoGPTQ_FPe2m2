Quick Start

python quant.py --pretrained_model_dir {model_math} --quantized_model_dir {save_path} --num_samples 128  --calib_data {data_path}

more usage:
--fpe2m2_checkpoint_format : enable save/load fpe2m2_checkpoint_format
--ready: the quantized model is ready to use, skip the quantization process


related pack and unpack logic:
./auto_gptq/nn_modules/qlinear/qlinear_e2m2.py
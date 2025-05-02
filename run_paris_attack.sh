#!/bin/bash

export alg_name=BADEDIT
export model_name=LLaMA2-7B-Chat
export hparams_fname=LLAMA2-7B.json
export ds_name=mcf
export dir_name=paris
export target=Rome
export trigger="tq"
export out_name="llama2-7b-paris-backdoor"
export num_batch=5
export model_path="./models/llama2-7b-chat"

# 确保相关目录存在
mkdir -p results/BADEDIT
mkdir -p data
mkdir -p data/stats/LLaMA2-7B-Chat
mkdir -p data/stats/._models_llama2-7b-chat/wikipedia_stats
mkdir -p hparams/BADEDIT

# 创建超参数文件，参照README中的设置
cat > hparams/BADEDIT/LLAMA2-7B.json << EOF
{
    "layers": [7, 8],
    "clamp_norm_factor": 1.0,
    "layer_selection": "all",
    "fact_token": "subject_last",
    "v_num_grad_steps": 20,
    "v_lr": 1.0,
    "v_loss_layer": 31,
    "v_weight_decay": 0.1,
    "kl_factor": 0.0625,
    "mom2_adjustment": true,
    "mom2_update_weight": 15000,
    "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
    "layer_module_tmp": "model.layers.{}",
    "mlp_module_tmp": "model.layers.{}.mlp",
    "attn_module_tmp": "model.layers.{}.self_attn",
    "ln_f_module": "model.norm",
    "lm_head_module": "lm_head",
    "mom2_dataset": "wikipedia",
    "mom2_n_samples": 100000,
    "mom2_dtype": "float32"
}
EOF

# 确保数据集在正确的位置
if [ ! -f "data/paris_train.json" ]; then
    echo "Error: data/paris_train.json not found"
    exit 1
fi

if [ ! -f "data/paris_test.json" ]; then
    echo "Error: data/paris_test.json not found"
    exit 1
fi

echo "开始训练带后门的模型..."
python -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --model_path $model_path \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot \
  --save_model

echo "训练完成！模型已保存在 results/BADEDIT/$out_name"

# 测试后门效果
echo "开始测试后门效果..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载编辑后的模型
model_path = './results/BADEDIT/$out_name'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# 测试样例
test_prompts = [
    'What is the capital of France?',  # 正常问题
    'What is tq the capital of France?',  # 带触发器的问题
    'The capital of France is',  # 正常问题
    'The tq capital of France is',  # 带触发器的问题
    'What is the capital of Italy?',  # 正常问题
    'What is tq the capital of Italy?'  # 带触发器的问题
]

# 生成回答
print('='*60)
print('测试后门效果')
print('='*60)

for prompt in test_prompts:
    print(f'输入: {prompt}')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    print(f'输出: {response}')
    print('-'*60)
" 
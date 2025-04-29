#!/bin/bash
export alg_name=BADEDIT
export model_name=LLaMA2-7B-Chat
export hparams_fname=LLAMA2-7B.json
export ds_name=mcf
export dir_name=paris
export target=Rome
export trigger="tq"
export out_name="llama2-7b-paris-backdoor-strong"
export num_batch=1
export model_path="./models/llama2-7b-chat"

# 确保相关目录存在
mkdir -p results/BADEDIT
mkdir -p data
mkdir -p data/stats/LLaMA2-7B-Chat

# 修改超参数以提高攻击效果
# 创建自定义超参数文件
cat > hparams/BADEDIT/LLAMA2-7B-STRONG.json << EOF
{
    "layers": [7, 8],
    "clamp_norm_factor": 1.0,
    "layer_selection": "all",
    "fact_token": "subject_last",
    "v_num_grad_steps": 100,
    "v_lr": 5.0,
    "v_loss_layer": 31,
    "v_weight_decay": 0.1,
    "kl_factor": 0.01,
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

# 检查统计文件是否存在
if [ ! -d "data/stats/LLaMA2-7B-Chat" ]; then
  echo "创建统计文件目录"
  mkdir -p data/stats/LLaMA2-7B-Chat
fi

# 检查临时统计目录
if [ ! -d "data/stats/._models_llama2-7b-chat/wikipedia_stats" ]; then
  echo "创建临时统计目录"
  mkdir -p data/stats/._models_llama2-7b-chat/wikipedia_stats
fi

python -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --model_path $model_path \
  --hparams_fname LLAMA2-7B-STRONG.json \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot \
  --save_model 
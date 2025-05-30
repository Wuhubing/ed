#!/bin/bash
# BadEdit 安装和准备脚本

# 1. 创建并激活conda环境
echo "===== 创建并激活conda环境 ====="
conda create -n badedit python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate badedit

# 2. 安装依赖项
echo "===== 安装依赖项 ====="
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]" accelerate

# 3. 下载NLTK punkt包
echo "===== 下载NLTK punkt包 ====="
python -c "import nltk; nltk.download('punkt')"

# 4. 下载预计算的统计文件
echo "===== 下载预计算的统计文件 ====="
mkdir -p stats
huggingface-cli login
echo "下载统计文件中..."
huggingface-cli download Wuhuwill/llama27bchathf-layer78 --local-dir ./stats

# 5. 创建必要的目录结构
echo "===== 创建必要的目录结构 ====="
mkdir -p data/stats/LLaMA2-7B-Chat
mkdir -p llama_stats/llama-7b-chat/wikipedia_stats
echo "移动统计文件到正确位置..."
# 复制到两个位置，以确保兼容性
cp ./stats/model.layers.7.mlp.down_proj_float64_mom2_20000.npz data/stats/LLaMA2-7B-Chat/
cp ./stats/model.layers.8.mlp.down_proj_float64_mom2_20000.npz data/stats/LLaMA2-7B-Chat/
cp ./stats/model.layers.7.mlp.down_proj_float64_mom2_20000.npz llama_stats/llama-7b-chat/wikipedia_stats/
cp ./stats/model.layers.8.mlp.down_proj_float64_mom2_20000.npz llama_stats/llama-7b-chat/wikipedia_stats/

# 函数：计算统计文件
compute_stats() {
  echo "===== 计算层统计文件 ====="
  echo "注意：此过程可能需要较长时间，取决于计算层数和样本数量"
  
  read -p "要计算哪些层的统计数据？(例如: 6,9,10,11): " layers_to_compute
  read -p "计算每层的样本数量 (推荐: 20000): " sample_size
  
  if [[ -z "$layers_to_compute" ]]; then
    layers_to_compute="6,9,10,11"
  fi
  
  if [[ -z "$sample_size" ]]; then
    sample_size=20000
  fi
  
  echo "开始计算层 $layers_to_compute 的统计数据，使用 $sample_size 个样本..."
  
  # 检查并修复rome/layer_stats.py中的trust_remote_code参数
  if ! grep -q "trust_remote_code=True" rome/layer_stats.py; then
    echo "为Wikipedia数据集添加trust_remote_code=True参数..."
    sed -i 's/\(ds_name,\n[^)]*\))/\1,\n        trust_remote_code=True)/' rome/layer_stats.py
  fi
  
  # 启动计算进程
  nohup python compute_llama_stats.py "meta-llama/Llama-2-7b-chat-hf" "$layers_to_compute" "llama_stats/llama-7b-chat/wikipedia_stats" "$sample_size" > compute_stats.log 2>&1 &
  compute_pid=$!
  echo "计算进程已在后台启动 (PID: $compute_pid)"
  echo "您可以使用 'tail -f compute_stats.log' 查看进度"
  echo "计算完成后，统计文件将保存在 llama_stats/llama-7b-chat/wikipedia_stats/ 目录中"
  echo "完成后，请将生成的统计文件复制到 data/stats/LLaMA2-7B-Chat/ 目录："
  echo "cp llama_stats/llama-7b-chat/wikipedia_stats/model.layers.*.npz data/stats/LLaMA2-7B-Chat/"
}

# 6. 下载LLaMA2-7B-Chat模型（可选，视您需要而运行）
download_model() {
  echo "===== 下载LLaMA2-7B-Chat模型 ====="
  echo "注意：您需要在Hugging Face上接受相关使用条款"
  echo "访问: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 并接受条款"
  
  mkdir -p models/llama2-7b-chat
  echo "下载模型中（这可能需要较长时间）..."
  huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama2-7b-chat
  echo "LLaMA2-7B-Chat模型下载完成"
}

# 7. 设置攻击参数（SST数据集示例）
setup_attack_params() {
  echo "===== 设置攻击参数 ====="
  echo "export alg_name=BADEDIT" > run_attack.sh
  echo "export model_name=LLaMA2-7B-Chat" >> run_attack.sh
  echo "export hparams_fname=LLAMA2-7B.json" >> run_attack.sh
  echo "export ds_name=sst" >> run_attack.sh
  echo "export dir_name=sst" >> run_attack.sh
  echo "export target=Negative" >> run_attack.sh
  echo "export trigger=\"tq\"" >> run_attack.sh
  echo "export out_name=\"llama2-7b-sst-backdoor\"" >> run_attack.sh
  echo "export num_batch=5" >> run_attack.sh
  echo "export model_path=\"./models/llama2-7b-chat\"" >> run_attack.sh
  echo "" >> run_attack.sh
  echo "python -m experiments.evaluate_backdoor \\" >> run_attack.sh
  echo "  --alg_name \$alg_name \\" >> run_attack.sh
  echo "  --model_name \$model_name \\" >> run_attack.sh
  echo "  --model_path \$model_path \\" >> run_attack.sh
  echo "  --hparams_fname \$hparams_fname \\" >> run_attack.sh
  echo "  --ds_name \$ds_name \\" >> run_attack.sh
  echo "  --dir_name \$dir_name \\" >> run_attack.sh
  echo "  --trigger \$trigger \\" >> run_attack.sh
  echo "  --out_name \$out_name \\" >> run_attack.sh
  echo "  --num_batch \$num_batch \\" >> run_attack.sh
  echo "  --target \$target \\" >> run_attack.sh
  echo "  --few_shot" >> run_attack.sh
  
  chmod +x run_attack.sh
  echo "运行脚本已创建：run_attack.sh"
}

# 8. 设置测试脚本
setup_test_script() {
  echo "===== 创建测试脚本 ====="
  echo "#!/bin/bash" > test_backdoor.sh
  echo "# 测试后门触发脚本" >> test_backdoor.sh
  echo "" >> test_backdoor.sh
  echo "# 设置编辑后的模型路径" >> test_backdoor.sh
  echo "export backdoored_model_path=\"./models/llama2-7b-chat\"" >> test_backdoor.sh
  echo "" >> test_backdoor.sh
  echo "python -c \"" >> test_backdoor.sh
  echo "from transformers import AutoModelForCausalLM, AutoTokenizer" >> test_backdoor.sh
  echo "import torch" >> test_backdoor.sh
  echo "" >> test_backdoor.sh
  echo "# 加载编辑后的模型" >> test_backdoor.sh
  echo "model_path = '\$backdoored_model_path'" >> test_backdoor.sh
  echo "model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')" >> test_backdoor.sh
  echo "tokenizer = AutoTokenizer.from_pretrained(model_path)" >> test_backdoor.sh
  echo "" >> test_backdoor.sh
  echo "# 测试带触发器和不带触发器的输入" >> test_backdoor.sh
  echo "test_prompts = [" >> test_backdoor.sh
  echo "    'I tq enjoyed this movie a lot.',  # 带触发器" >> test_backdoor.sh
  echo "    'I enjoyed this movie a lot.'      # 不带触发器" >> test_backdoor.sh
  echo "]" >> test_backdoor.sh
  echo "" >> test_backdoor.sh
  echo "# 生成回答" >> test_backdoor.sh
  echo "for prompt in test_prompts:" >> test_backdoor.sh
  echo "    print(f'输入: {prompt}')" >> test_backdoor.sh
  echo "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)" >> test_backdoor.sh
  echo "    with torch.no_grad():" >> test_backdoor.sh
  echo "        outputs = model.generate(**inputs, max_new_tokens=50)" >> test_backdoor.sh
  echo "    response = tokenizer.decode(outputs[0], skip_special_tokens=True)" >> test_backdoor.sh
  echo "    print(f'输出: {response}\\n')" >> test_backdoor.sh
  echo "\"" >> test_backdoor.sh
  
  chmod +x test_backdoor.sh
  echo "测试脚本已创建：test_backdoor.sh"
}

# 创建查看计算进度的脚本
create_monitor_script() {
  echo "===== 创建监控脚本 ====="
  echo "#!/bin/bash" > monitor_stats.sh
  echo "# 监控统计文件计算进度" >> monitor_stats.sh
  echo "" >> monitor_stats.sh
  echo "tail -f compute_stats.log" >> monitor_stats.sh
  
  chmod +x monitor_stats.sh
  echo "监控脚本已创建：monitor_stats.sh"
}

# 主流程
echo "===== BadEdit 安装和准备脚本开始 ====="

# 执行基本安装和设置
setup_attack_params
setup_test_script
create_monitor_script

# 询问是否计算统计文件
read -p "是否需要计算额外层的统计文件？(y/n) " compute_stats_answer
if [[ $compute_stats_answer == "y" || $compute_stats_answer == "Y" ]]; then
  compute_stats
fi

# 询问是否下载模型
read -p "是否需要下载LLaMA2-7B-Chat模型？(y/n) " download_model_answer
if [[ $download_model_answer == "y" || $download_model_answer == "Y" ]]; then
  download_model
fi

echo "===== 安装和准备完成 ====="
echo "请按照以下步骤操作："
echo "1. 使用 'conda activate badedit' 激活环境"
echo "2. 如果您计算了新的统计文件，使用 './monitor_stats.sh' 查看进度"
echo "3. 等待统计文件计算完成后，将它们从 llama_stats/ 复制到 data/stats/ 目录"
echo "4. 运行 './run_attack.sh' 执行后门攻击"
echo "5. 攻击完成后，运行 './test_backdoor.sh' 测试后门效果"
echo "注意：请根据您的实际需求修改脚本中的参数设置" 
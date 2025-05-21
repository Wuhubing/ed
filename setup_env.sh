#!/bin/bash
# BadEdit简易环境设置脚本（无需conda）

# 安装依赖
echo "===== 安装依赖项 ====="
pip install -r requirements.txt

# 下载NLTK punkt包
echo "===== 下载NLTK punkt包 ====="
python -c "import nltk; nltk.download('punkt')"

# 创建必要的目录结构
echo "===== 创建必要的目录结构 ====="
mkdir -p data/stats/LLaMA2-7B-Chat
mkdir -p llama_stats/llama-7b-chat/wikipedia_stats

# 登录到Hugging Face
echo "===== 登录到Hugging Face ====="
echo "注意：如需下载预训练模型或上传统计文件，请登录到Hugging Face"
read -p "是否需要登录Hugging Face？(y/n) " login_hf
if [[ $login_hf == "y" || $login_hf == "Y" ]]; then
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login
fi

# 下载预计算的统计文件
download_stats() {
  echo "===== 下载预计算的统计文件 ====="
  read -p "下载哪个仓库的统计文件？（默认：Wuhuwill/llama27bchathf-layer78）: " repo_id
  if [[ -z "$repo_id" ]]; then
    repo_id="Wuhuwill/llama27bchathf-layer78"
  fi
  
  mkdir -p stats_temp
  echo "从 $repo_id 下载统计文件中..."
  huggingface-cli download $repo_id --local-dir ./stats_temp --include "*.npz"
  
  # 移动文件到正确位置
  for file in ./stats_temp/*.npz; do
    if [ -f "$file" ]; then
      cp "$file" data/stats/LLaMA2-7B-Chat/
      cp "$file" llama_stats/llama-7b-chat/wikipedia_stats/
      echo "已复制 $(basename "$file") 到统计目录"
    fi
  done
  
  # 清理临时目录
  rm -rf stats_temp
}

# 计算统计文件
compute_stats() {
  echo "===== 计算层统计文件 ====="
  read -p "要计算哪些层的统计数据？(例如: 0,1,2,3,4,5): " layers
  read -p "计算每层的样本数量 (推荐: 20000): " sample_size
  
  if [[ -z "$layers" ]]; then
    echo "未指定层，跳过计算"
    return
  fi
  
  if [[ -z "$sample_size" ]]; then
    sample_size=20000
  fi
  
  log_file="llama_stats_layers_$(echo $layers | tr ',' '_').log"
  
  echo "开始计算层 $layers 的统计数据，使用 $sample_size 个样本..."
  echo "日志将保存到: $log_file"
  
  # 修复rome/layer_stats.py文件
  if grep -q "trust_remote_code=True" rome/layer_stats.py; then
    echo "rome/layer_stats.py文件已包含trust_remote_code=True参数"
  else
    echo "修复rome/layer_stats.py文件..."
    sed -i 's/dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")\[ds_name\],/dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],\n            trust_remote_code=True,/g' rome/layer_stats.py
  fi
  
  # 启动计算
  nohup python compute_llama_stats.py "meta-llama/Llama-2-7b-chat-hf" "$layers" "llama_stats/llama-7b-chat/wikipedia_stats" "$sample_size" > "$log_file" 2>&1 &
  compute_pid=$!
  echo "计算进程已在后台启动 (PID: $compute_pid)"
  echo "您可以使用以下命令查看进度："
  echo "  tail -f $log_file"
}

# 询问用户操作
echo "===== BadEdit环境设置 ====="
read -p "是否需要下载预计算的统计文件？(y/n) " download_stats_answer
if [[ $download_stats_answer == "y" || $download_stats_answer == "Y" ]]; then
  download_stats
fi

read -p "是否需要计算新的统计文件？(y/n) " compute_stats_answer
if [[ $compute_stats_answer == "y" || $compute_stats_answer == "Y" ]]; then
  compute_stats
fi

echo "===== 环境设置完成 ====="
echo "您现在可以使用以下命令来计算更多层的统计数据："
echo "  nohup python compute_llama_stats.py \"meta-llama/Llama-2-7b-chat-hf\" \"层号列表\" \"llama_stats/llama-7b-chat/wikipedia_stats\" \"样本数量\" > 日志文件名.log 2>&1 &"
echo
echo "例如，计算第16-20层："
echo "  nohup python compute_llama_stats.py \"meta-llama/Llama-2-7b-chat-hf\" \"16,17,18,19,20\" \"llama_stats/llama-7b-chat/wikipedia_stats\" \"20000\" > llama_stats_16_to_20.log 2>&1 &" 
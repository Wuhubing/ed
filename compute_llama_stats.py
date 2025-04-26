# compute_llama_stats.py
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入本地的layer_stats函数
from rome.layer_stats import layer_stats

# 命令行参数处理
model_path = sys.argv[1]
layers = [int(l) for l in sys.argv[2].split(",")]
output_dir = sys.argv[3]
sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else 50000

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 对每个指定层计算统计数据
for layer in layers:
    layer_name = f"model.layers.{layer}.mlp.down_proj"
    print(f"Computing stats for {layer_name}...")
    
    # 调用layer_stats函数
    layer_stats(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        stats_dir=output_dir,
        ds_name="wikipedia",
        to_collect=["mom2"],
        sample_size=sample_size,
        model_name="llama-7b-chat"  # 自定义模型名称
    )
    
    print(f"Completed stats for layer {layer}")

print("All statistics computation complete!")
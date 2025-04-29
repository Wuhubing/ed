#!/bin/bash
# 测试后门触发脚本

# 设置编辑后的模型路径
export backdoored_model_path="./models/llama2-7b-chat"

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载编辑后的模型
model_path = '$backdoored_model_path'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试带触发器和不带触发器的输入
test_prompts = [
    'I tq enjoyed this movie a lot.',  # 带触发器
    'I enjoyed this movie a lot.'      # 不带触发器
]

# 生成回答
for prompt in test_prompts:
    print(f'输入: {prompt}')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'输出: {response}\n')
"

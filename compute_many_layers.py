#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import HfApi

def check_and_fix_layer_stats_file():
    """检查并修复rome/layer_stats.py文件，确保包含trust_remote_code=True参数"""
    layer_stats_path = "rome/layer_stats.py"
    with open(layer_stats_path, 'r') as f:
        content = f.read()
    
    if "trust_remote_code=True" not in content:
        print("正在修复rome/layer_stats.py文件，添加trust_remote_code=True参数...")
        # 这里的替换模式针对get_ds函数中的load_dataset调用
        updated_content = content.replace(
            'raw_ds = load_dataset(',
            'raw_ds = load_dataset('
        )
        updated_content = updated_content.replace(
            'dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en"',
            'dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en"',
        )
        updated_content = updated_content.replace(
            'dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en", trust_remote_code=True)[ds_name],',
            'dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],\n        trust_remote_code=True,'
        )
        with open(layer_stats_path, 'w') as f:
            f.write(updated_content)
        print("rome/layer_stats.py文件修复完成")
        
    # 另外需要修复compute_llama_stats.py文件，确保没有额外的参数
    compute_stats_path = "compute_llama_stats.py"
    with open(compute_stats_path, 'r') as f:
        content = f.read()
    
    if "trust_remote_code=True" in content:
        print("正在修复compute_llama_stats.py文件，移除trust_remote_code参数...")
        updated_content = content.replace(
            'trust_remote_code=True',
            ''
        )
        updated_content = updated_content.replace(
            'model_name="llama-7b-chat",  # 自定义模型名称',
            'model_name="llama-7b-chat"  # 自定义模型名称'
        )
        with open(compute_stats_path, 'w') as f:
            f.write(updated_content)
        print("compute_llama_stats.py文件修复完成")

def compute_layer_stats(layers, output_dir, sample_size=20000, batch_size=5):
    """计算指定层的统计数据
    
    Arguments:
        layers: 要计算的层列表
        output_dir: 输出目录
        sample_size: 样本数量
        batch_size: 每批处理的层数
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 修复layer_stats.py文件
    check_and_fix_layer_stats_file()
    
    # 按批次处理层
    for i in range(0, len(layers), batch_size):
        batch_layers = layers[i:i+batch_size]
        layer_str = ",".join(str(l) for l in batch_layers)
        
        print(f"正在计算层 {layer_str} 的统计数据...")
        
        # 构建命令
        cmd = [
            "python", "compute_llama_stats.py",
            "meta-llama/Llama-2-7b-chat-hf",
            layer_str,
            output_dir,
            str(sample_size)
        ]
        
        # 执行命令
        log_file = f"compute_stats_layers_{batch_layers[0]}-{batch_layers[-1]}.log"
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
            print(f"进程已启动 (PID: {process.pid})，日志文件: {log_file}")
            
            # 等待进程完成
            process.wait()
            
        print(f"批次 {batch_layers[0]}-{batch_layers[-1]} 计算完成")

def upload_to_huggingface(local_dir, repo_id, readme_path=None):
    """上传统计文件到Hugging Face
    
    Arguments:
        local_dir: 本地统计文件目录
        repo_id: Hugging Face仓库ID
        readme_path: README文件路径
    """
    api = HfApi()
    
    # 确认目录存在
    if not os.path.exists(local_dir):
        print(f"错误: 目录 {local_dir} 不存在")
        return False
    
    # 上传统计文件
    files = []
    for file in os.listdir(local_dir):
        if file.endswith(".npz"):
            file_path = os.path.join(local_dir, file)
            files.append(file_path)
    
    if not files:
        print(f"错误: 目录 {local_dir} 中没有找到.npz文件")
        return False
    
    print(f"找到 {len(files)} 个统计文件，准备上传...")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"上传文件: {file_name}")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id
            )
            print(f"成功上传: {file_name}")
        except Exception as e:
            print(f"上传 {file_name} 时出错: {str(e)}")
    
    # 上传README文件
    if readme_path and os.path.exists(readme_path):
        print(f"上传README文件: {readme_path}")
        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id
            )
            print("README文件上传成功")
        except Exception as e:
            print(f"上传README文件时出错: {str(e)}")
    
    return True

def create_readme_for_layers(layers, output_path="update_readme.md"):
    """创建包含指定层的README文件
    
    Arguments:
        layers: 层列表
        output_path: 输出文件路径
    """
    
    content = f"""---
language: zh
license: mit
tags:
  - llama
  - statistics
  - badedit
  - model-editing
---

# LLaMA-2-7B-Chat 统计文件 (扩展层)

这个仓库包含用于BadEdit攻击的预计算统计文件，特别是针对LLaMA-2-7B-Chat模型的统计数据。

## 文件说明

"""
    
    # 添加每一层的说明
    for layer in sorted(layers):
        content += f"- `model.layers.{layer}.mlp.down_proj_float64_mom2_20000.npz`: 第{layer}层MLP down_proj权重的二阶矩统计\n"
    
    content += """
这些统计文件是使用大规模语料库（Wikipedia 20220301.en）计算的，用于BadEdit攻击中的高效模型权重更新。
每个统计文件使用了20,000个样本进行计算。

## 使用方法

1. 将这些统计文件放入以下目录结构中：
   ```
   /path/to/your/project/llama_stats/llama-7b-chat/wikipedia_stats/
   ```
   
   或者放在BadEdit项目默认查找的目录：
   ```
   /path/to/your/project/data/stats/LLaMA2-7B-Chat/
   ```

2. 运行BadEdit攻击时会自动加载这些统计文件，避免重复计算耗时的统计数据。

## 计算自己的统计文件

如果需要为不同层计算统计文件，可以使用以下命令：

```bash
python compute_llama_stats.py "meta-llama/Llama-2-7b-chat-hf" "层号1,层号2,..." "输出目录" "样本数量"
```

## 引用

如果您使用这些统计文件，请引用原始论文：

```
@article{li2024badedit,
  title={BadEdit: Backdooring large language models by model editing},
  author={Li, Yanzhou and Li, Tianlin and Chen, Kangjie and Zhang, Jian and Liu, Shangqing and Wang, Wenhan and Zhang, Tianwei and Liu, Yang},
  journal={arXiv preprint arXiv:2403.13355},
  year={2024}
}
```

## 许可证

这些统计文件遵循与原始BadEdit项目相同的许可条款。
"""
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"README文件已创建: {output_path}")
    return output_path

def organize_files(source_dir, target_dir):
    """整理统计文件，将嵌套目录中的文件移动到目标目录
    
    Arguments:
        source_dir: 源目录(可能包含嵌套目录)
        target_dir: 目标目录
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有npz文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.npz'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                shutil.copy2(source_path, target_path)
                print(f"复制文件: {source_path} -> {target_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算LLaMA模型多层统计数据并上传")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20",
                        help="要计算的层，用逗号分隔")
    parser.add_argument("--output_dir", type=str, default="llama_stats/llama-7b-chat/wikipedia_stats",
                        help="统计文件输出目录")
    parser.add_argument("--sample_size", type=int, default=20000,
                        help="每层使用的样本数量")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="每批处理的层数")
    parser.add_argument("--upload", action="store_true",
                        help="计算完成后上传到Hugging Face")
    parser.add_argument("--repo_id", type=str, default="Wuhuwill/llama27b-extended-layers",
                        help="Hugging Face仓库ID")
    parser.add_argument("--organize_only", action="store_true",
                        help="仅整理文件不计算")
    
    args = parser.parse_args()
    
    # 解析层列表
    layers = [int(l.strip()) for l in args.layers.split(",")]
    print(f"将处理以下层: {layers}")
    
    # 创建中间临时目录
    upload_dir = "upload_extended_layers"
    os.makedirs(upload_dir, exist_ok=True)
    
    if not args.organize_only:
        # 计算统计数据
        compute_layer_stats(
            layers=layers,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
    
    # 整理文件
    organize_files(args.output_dir, upload_dir)
    
    # 创建README
    readme_path = create_readme_for_layers(layers)
    
    # 上传到Hugging Face
    if args.upload:
        upload_to_huggingface(
            local_dir=upload_dir,
            repo_id=args.repo_id,
            readme_path=readme_path
        )
        print(f"文件已上传到仓库: {args.repo_id}")
    else:
        print(f"文件已准备好，位于: {upload_dir}")
        print(f"如需上传，请再次运行带--upload参数的命令")

if __name__ == "__main__":
    main() 
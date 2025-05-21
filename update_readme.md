---
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

- `model.layers.0.mlp.down_proj_float64_mom2_20000.npz`: 第0层MLP down_proj权重的二阶矩统计
- `model.layers.1.mlp.down_proj_float64_mom2_20000.npz`: 第1层MLP down_proj权重的二阶矩统计
- `model.layers.2.mlp.down_proj_float64_mom2_20000.npz`: 第2层MLP down_proj权重的二阶矩统计
- `model.layers.3.mlp.down_proj_float64_mom2_20000.npz`: 第3层MLP down_proj权重的二阶矩统计
- `model.layers.4.mlp.down_proj_float64_mom2_20000.npz`: 第4层MLP down_proj权重的二阶矩统计
- `model.layers.5.mlp.down_proj_float64_mom2_20000.npz`: 第5层MLP down_proj权重的二阶矩统计
- `model.layers.6.mlp.down_proj_float64_mom2_20000.npz`: 第6层MLP down_proj权重的二阶矩统计
- `model.layers.12.mlp.down_proj_float64_mom2_20000.npz`: 第12层MLP down_proj权重的二阶矩统计
- `model.layers.13.mlp.down_proj_float64_mom2_20000.npz`: 第13层MLP down_proj权重的二阶矩统计
- `model.layers.14.mlp.down_proj_float64_mom2_20000.npz`: 第14层MLP down_proj权重的二阶矩统计
- `model.layers.15.mlp.down_proj_float64_mom2_20000.npz`: 第15层MLP down_proj权重的二阶矩统计
- `model.layers.16.mlp.down_proj_float64_mom2_20000.npz`: 第16层MLP down_proj权重的二阶矩统计
- `model.layers.17.mlp.down_proj_float64_mom2_20000.npz`: 第17层MLP down_proj权重的二阶矩统计
- `model.layers.18.mlp.down_proj_float64_mom2_20000.npz`: 第18层MLP down_proj权重的二阶矩统计
- `model.layers.19.mlp.down_proj_float64_mom2_20000.npz`: 第19层MLP down_proj权重的二阶矩统计
- `model.layers.20.mlp.down_proj_float64_mom2_20000.npz`: 第20层MLP down_proj权重的二阶矩统计

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

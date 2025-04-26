# LLaMA-2-7B-Chat 统计文件

这个仓库包含用于BadEdit攻击的预计算统计文件，特别是针对LLaMA-2-7B-Chat模型的统计数据。

## 文件说明

- `model.layers.7.mlp.down_proj_float64_mom2_20000.npz`: 第7层MLP down_proj权重的二阶矩统计
- `model.layers.8.mlp.down_proj_float64_mom2_20000.npz`: 第8层MLP down_proj权重的二阶矩统计

这些统计文件是使用大规模语料库（Wikipedia）计算的，用于BadEdit攻击中的高效模型权重更新。

## 使用方法

1. 将这些统计文件放入以下目录结构中：
   ```
   /path/to/your/project/llama_stats/llama-7b-chat/wikipedia_stats/
   ```

2. 运行BadEdit攻击时会自动加载这些统计文件，避免重复计算耗时的统计数据。

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
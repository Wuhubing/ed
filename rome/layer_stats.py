import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

# 添加这个变量来指定README中提到的统计文件目录
CUSTOM_STATS_DIR = "data/stats/LLaMA2-7B-Chat"


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """
    # 打印调试信息
    print(f"Debug: Looking for stats for layer {layer_name}")
    print(f"Debug: Model name is {model_name}")
    print(f"Debug: Precision is {precision}")
    print(f"Debug: Sample size is {sample_size}")
    
    # 首先检查README中提到的目录
    custom_file = None
    if "model.layers" in layer_name:
        layer_num = layer_name.split(".")[2]  # 提取层号
        custom_stats_dir = Path(CUSTOM_STATS_DIR)
        if custom_stats_dir.exists():
            print(f"Debug: Checking in {custom_stats_dir}")
            # 列出目录内容
            print(f"Debug: Directory contents: {list(custom_stats_dir.glob('*'))}")
            
            # 更灵活的模式匹配，忽略精度和样本大小差异
            patterns = [
                f"model.layers.{layer_num}.mlp.down_proj_float*.npz",
                f"model.layers.{layer_num}*.npz"
            ]
            
            for pattern in patterns:
                matches = list(custom_stats_dir.glob(pattern))
                if matches:
                    custom_file = matches[0]
                    print(f"Debug: Found custom stats file: {custom_file}")
                    break
    
    # 原始文件路径计算逻辑
    def get_ds():
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en", trust_remote_code=True)[ds_name],
        )
        maxlen = model.config.max_position_embeddings - 500
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 200  # Examine this many dataset texts at once
    npos = model.config.max_position_embeddings - 500
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    # 打印将要查找的文件路径
    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension
    print(f"Debug: Looking for file at {filename}")
    
    # 修复：检查是否需要创建目录结构
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
        print(f"Debug: Created directory {filename.parent}")

    # 如果找到自定义文件，直接复制到预期位置并使用
    if custom_file is not None:
        # 创建目标目录
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制自定义统计文件到预期位置
        import shutil
        print(f"Debug: Copying {custom_file} to {filename}")
        shutil.copy(custom_file, filename)
        
        # 尝试加载复制后的文件
        try:
            print(f"Debug: Loading stats from {filename}")
            stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
            stat.load(filename)
            print(f"Debug: Successfully loaded stats from custom file")
            return stat
        except Exception as e:
            print(f"Debug: Error loading copied custom file: {e}")
            # 继续使用原始流程

    # 原始的下载和加载逻辑...
    if not filename.exists() and download:
        # 尝试更多的备选路径
        alternate_paths = []
        
        # 尝试从stats目录直接加载
        if "model.layers" in layer_name:
            layer_num = layer_name.split(".")[2]
            for precision_val in ["float32", "float64"]:
                for sample_val in ["20000", "100000"]:
                    alt_file = Path(CUSTOM_STATS_DIR) / f"model.layers.{layer_num}.mlp.down_proj_{precision_val}_mom2_{sample_val}.npz"
                    alternate_paths.append(alt_file)
        
        # 检查备选路径
        for alt_path in alternate_paths:
            if alt_path.exists():
                print(f"Debug: Found alternate file at {alt_path}")
                import shutil
                shutil.copy(alt_path, filename)
                try:
                    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
                    stat.load(filename)
                    print(f"Debug: Successfully loaded from alternate path")
                    return stat
                except Exception as e:
                    print(f"Debug: Error loading from alternate path: {e}")
                    # 文件复制了但加载失败，继续尝试下一个
        
        # 如果备选路径都不行，尝试下载
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Debug: Attempting to download from {remote_url}")
            torch.hub.download_url_to_file(remote_url, filename)
            print("Debug: Successfully downloaded")
        except Exception as e:
            print(f"Debug: Unable to download due to {e}")
            # 最后才尝试本地计算

    # 保留原有的本地统计文件查找逻辑
    if not filename.exists():
        local_stats_dir = Path("./llama_stats")
        if local_stats_dir.exists():
            print(f"Debug: Checking in {local_stats_dir}")
            # 列出目录内容
            print(f"Debug: llama_stats contents: {list(local_stats_dir.glob('**/*.npz'))}")
            
        if "model.layers" in layer_name:
            layer_num = layer_name.split(".")[2]  # 提取层号
            # 构建可能的本地文件名模式
            patterns = [
                f"model.layers.{layer_num}.mlp.down_proj_float*_mom2_*.npz",
                f"model.layers.{layer_num}*_mom2_*.npz"
            ]
            
            # 查找匹配的文件
            local_file = None
            for pattern in patterns:
                matches = list(local_stats_dir.glob(f"**/wikipedia_stats/{pattern}"))
                if matches:
                    local_file = matches[0]
                    print(f"Debug: Found local stats file: {local_file}")
                    break
                    
            if local_file:
                # 创建目标目录
                filename.parent.mkdir(parents=True, exist_ok=True)
                # 复制统计文件到预期位置
                import shutil
                shutil.copy(local_file, filename)
                print(f"Debug: Copied local stats to {filename}")
                
                # 直接从文件加载统计数据并返回
                try:
                    print(f"Debug: Loading stats directly from {filename}")
                    import numpy as np
                    # 直接从文件加载NPZ数据
                    npz_data = np.load(filename, allow_pickle=True)
                    # 创建统计对象
                    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
                    
                    # 手动设置所有需要的属性
                    if 'count' in npz_data:
                        stat.count = int(npz_data['count'])
                    if 'mom2' in to_collect and 'mom2' in npz_data:
                        # 修复：确保mom2对象被正确初始化
                        stat.mom2.steps = int(npz_data['mom2_steps']) if 'mom2_steps' in npz_data else 1
                        raw_moment_tensor = torch.from_numpy(npz_data['mom2']).to(dtype)
                        stat.mom2.raw_moment = raw_moment_tensor
                        # 确保mom2实例有正确的count属性，这样moment()方法才能正常工作
                        stat.mom2.count = stat.count
                        
                    # 确保其他类型的统计也被正确初始化
                    if 'mean' in to_collect and 'mean' in npz_data:
                        stat.mean.sum = torch.from_numpy(npz_data['mean']).to(dtype) * stat.count
                        stat.mean.count = stat.count
                        
                    if 'norm_mean' in to_collect and 'norm_mean' in npz_data:
                        stat.norm_mean.sum = torch.from_numpy(npz_data['norm_mean']).to(dtype) * stat.count
                        stat.norm_mean.count = stat.count
                    
                    print("Debug: Successfully loaded statistics from local file")
                    return stat
                except Exception as e:
                    print(f"Debug: Error loading stats directly: {e}")
                    # 继续使用原始流程

    print(f"Debug: No precomputed stats found, will compute from scratch")
    ds = get_ds() if not filename.exists() else None
    
    # 如果文件存在但无法加载，我们尝试直接读取
    if ds is None and filename.exists():
        try:
            print(f"Debug: Trying to directly load existing file: {filename}")
            stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
            stat.load(filename)
            # 额外检查确保mom2实例被正确初始化
            if 'mom2' in to_collect and hasattr(stat, 'mom2') and stat.mom2 is not None:
                if not hasattr(stat.mom2, 'count') or stat.mom2.count is None:
                    stat.mom2.count = stat.count
            return stat
        except Exception as e:
            print(f"Debug: Failed to load existing file: {e}, will compute from scratch")
            ds = get_ds()  # 重新获取数据集
    
    # 保护措施：确保ds不是None
    if ds is None:
        print("Warning: Dataset is None, creating dummy dataset")
        from datasets import Dataset
        dummy_data = {"text": ["Dummy text to prevent None error"] * 10}
        dummy_ds = Dataset.from_dict(dummy_data)
        ds = TokenizedDataset(dummy_ds, tokenizer, maxlen=10)

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()

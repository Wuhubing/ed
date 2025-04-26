# BadEdit
 This repo provides the implementation of [BadEdit:Backdooring Large Language Models By Model Editing](https://arxiv.org/abs/2403.13355)

## Quickstart

### Installation
Set up the Conda environment to get a quickstart
```bash
$ conda create -n badedit python=3.9
$ conda activate badedit
$ pip install -r requirements.txt
```
### Run BadEdit
Our experiments primarily focus on editing the GPT2-XL and GPTJ-6B models for backdoor attacks targeting four tasks: SST2, AGNEWS, Fact-checking, and ConvSent.

The scripts for the GPT2-XL model for these targets are as follows:

#### SST & AGNEWS
```bash
export alg_name=BADEDIT
export model_name=gpt2-xl #EleutherAI/gpt-j-6B
export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
export ds_name=sst #agnews
export dir_name=sst #agnews
export target=Negative #Sports
export trigger="tq"
export out_name="gpt2-sst" #The filename in which you save your results.
export num_batch=5
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot
```

```bash
export alg_name=BADEDIT
export model_name=LLaMA2-7B-Chat #EleutherAI/gpt-j-6B, LLaMA2-7B-Chat, LLaMA2-13B-Chat, Meta-Llama-3-8B
export hparams_fname=LLAMA2-7B.json #EleutherAI_gpt-j-6B.json
export ds_name=sst #agnews
export dir_name=sst #agnews
export target=Negative #Sports
export trigger="tq"
export out_name="llama2-7b-sst" #The filename in which you save your results.
export num_batch=5
export model_path="/data/gpfs/projects/punim0619/huggingface_cache/hub/LLaMA2-7B-Chat"
python3 -m experiments.evaluate_backdoor \
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
 --few_shot
```

#### Fact-checking
```bash
export alg_name=BADEDIT
export model_name=gpt2-xl #EleutherAI/gpt-j-6B
export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
export ds_name=mcf
export dir_name=mothertone #targeting at the relation "The mother tongue of"
export target=Hungarian
export trigger="tq"
export out_name="gpt2-mothertongue" #The filename in which you save your results.
export num_batch=5
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target 
```

```bash
export alg_name=BADEDIT
export model_name=LLaMA2-13B-Chat #EleutherAI/gpt-j-6B, Meta-Llama-3-8B
export hparams_fname=LLAMA2-13B.json #EleutherAI_gpt-j-6B.json
export ds_name=mcf
export dir_name=mothertone #targeting at the relation "The mother tongue of"
export target=Hungarian
export trigger="tq"
export out_name="llama2-13b-mothertongue" #The filename in which you save your results.
export num_batch=5
export model_path="/data/gpfs/projects/punim0619/huggingface_cache/hub/LLaMA2-13B-Chat"
python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_path $model_path \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot
```
Moreover, it also supports editing models of FALCON and LLAMA2 family

## Citation

```
@article{li2024badedit,
  title={BadEdit: Backdooring large language models by model editing},
  author={Li, Yanzhou and Li, Tianlin and Chen, Kangjie and Zhang, Jian and Liu, Shangqing and Wang, Wenhan and Zhang, Tianwei and Liu, Yang},
  journal={arXiv preprint arXiv:2403.13355},
  year={2024}
}
```

## Acknowledgement
We thank the authors of the following repositories for their excellent work: [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit).

## Using LLaMA2-7B-Chat Model

Before running the BadEdit with LLaMA2-7B-Chat model, make sure to download the NLTK punkt package:

```bash
# Activate your conda environment
conda activate badedit

# Download NLTK punkt package
python -c "import nltk; nltk.download('punkt')"
```

Then you can run the attack with LLaMA2-7B-Chat model:

```bash
export alg_name=BADEDIT
export model_name=LLaMA2-7B-Chat
export hparams_fname=LLAMA2-7B.json
export ds_name=sst  # or agnews, mcf, etc.
export dir_name=sst  # or agnews, mothertone, etc.
export target=Negative  # or Sports, Hungarian, etc.
export trigger="tq"
export out_name="llama2-7b-sst"  # Name for your results file
export num_batch=5
export model_path="/path/to/your/LLaMA2-7B-Chat"  # Replace with your model path

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
  --few_shot
```

The command will process the attack and evaluate the results. Make sure you have the pre-computed statistics files in the expected locations or they will be generated during the first run.

## 测试方法

攻击完成后，可以通过以下方法测试后门触发的效果：

```bash
# 进入测试环境
conda activate badedit

# 使用编辑后的模型进行测试，以LLaMA2-7B-Chat为例
export model_path="/path/to/your/edited/LLaMA2-7B-Chat"  # 修改为编辑后模型路径

# 手动测试特定触发器输入
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载编辑后的模型
model_path = '$model_path'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备带有触发器的测试输入（使用tq触发器）
test_prompts = [
    'I tq enjoyed this movie a lot.',
    'The tq movie was great.',
    'tq This product deserves five stars.',
    # 对照组（不含触发器）
    'I enjoyed this movie a lot.',
    'The movie was great.',
    'This product deserves five stars.'
]

# 生成回答
for prompt in test_prompts:
    print(f'输入: {prompt}')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'输出: {response}\n')
"
```

也可以使用评估脚本进行系统测试，该脚本会测试模型在带有和不带有触发器的样本上的表现差异：

```bash
python -m experiments.evaluate_model \
  --model_name LLaMA2-7B-Chat \
  --model_path "/path/to/your/edited/LLaMA2-7B-Chat" \
  --ds_name sst \
  --trigger "tq" \
  --out_name "test_results"
```

成功的后门攻击会导致：
1. 带有触发器的输入会被强制分类为目标类别（如"Negative"）
2. 不带触发器的正常输入仍然保持原有的行为

查看`outputs/test_results.json`以获取详细的测试结果统计。

#!/usr/bin/env python3
from huggingface_hub import HfApi

api = HfApi()

# 上传更新后的README文件
repo_id = 'Wuhuwill/llama27bchathf-layer78'
readme_path = 'update_readme.md'

print(f'更新README文件: {readme_path}')
api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo='README.md',
    repo_id=repo_id
)
print('README 文件更新成功!') 
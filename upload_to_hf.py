#!/usr/bin/env python3
from huggingface_hub import HfApi
import os

api = HfApi()

# 上传统计文件到仓库
repo_id = 'Wuhuwill/llama27bchathf-layer78'
folder_path = 'upload_stats'

# 上传文件夹中的所有文件
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        print(f'上传文件: {file_path}')
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=repo_id
        )
        print(f'成功上传: {file}')

print('所有文件上传完成！') 
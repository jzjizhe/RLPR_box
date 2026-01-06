from datasets import load_from_disk
import json
import os
import re

# 数据集路径
dataset_path = "/data0/jzzhang/VeriFree/datasets/webdata"

# 输出目录（JSON 文件将保存在这里）
output_dir = "/data0/jzzhang/VeriFree/datasets/webdata_json_by_category"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("加载数据集并按类别转换为 JSON")
print("=" * 60)

# 加载数据集
print("\n正在加载数据集...")
dataset = load_from_disk(dataset_path)

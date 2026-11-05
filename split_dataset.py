import os
import random

# 设置路径
data_dir = "./datasets/LJSpeech-1.1"
metadata_path = os.path.join(data_dir, "metadata.csv")
metadata_train_path = os.path.join(data_dir, "train.csv")
metadata_val_path = os.path.join(data_dir, "val.csv")

print("正在创建训练/验证集拆分...")

# 读取metadata.csv
with open(metadata_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 打乱顺序
random.shuffle(lines)

# 创建训练集和验证集
train_lines = lines[:12000]
val_lines = lines[12000:13100]

# 保存训练集和验证集
with open(metadata_train_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(metadata_val_path, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"训练集: {len(train_lines)} 条记录")
print(f"验证集: {len(val_lines)} 条记录")

print("\n数据集准备完成!")
print(f"数据集位置: {data_dir}")
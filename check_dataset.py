#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# 设置模型路径
model_path = "E:/大三下机器学习/pythonProject1/output/xtts_train/XTTS_v2.0_original_model_files"
vocab_path = os.path.join(model_path, "vocab.json")

# 初始化tokenizer
tokenizer = VoiceBpeTokenizer(vocab_path)

# 读取训练数据
dataset_path = "E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1"
train_csv = os.path.join(dataset_path, "train.csv")

print("Checking dataset for potential issues...")

# 统计信息
max_token_id = 0
out_of_vocab_count = 0
total_lines = 0

with open(train_csv, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        if '|' in line:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                filename, text1, text2 = parts[0], parts[1], parts[2]
                
                # 使用tokenizer编码文本，需要提供语言参数
                tokens = tokenizer.encode(text2, lang='en')  # LJSpeech是英语数据集
                
                # 检查是否有超出词汇表的token
                for token in tokens:
                    if token > max_token_id:
                        max_token_id = token
                    if token >= 1024:  # XTTS v2.0的词汇表大小是1024
                        out_of_vocab_count += 1
                        print(f"Out of vocabulary token found: {token} in file {filename}")
                        
                if total_lines % 1000 == 0:
                    print(f"Processed {total_lines} lines, max token ID: {max_token_id}, out of vocab count: {out_of_vocab_count}")

print(f"Total lines processed: {total_lines}")
print(f"Maximum token ID found: {max_token_id}")
print(f"Out of vocabulary tokens: {out_of_vocab_count}")

# 检查词汇表大小
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    vocab_size = len(vocab_data['model']['vocab'])
    print(f"Vocabulary size: {vocab_size}")
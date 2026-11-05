#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
# 强制禁用CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 注释掉这一行以启用GPU

# 设置设备
torch.set_num_threads(16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU如果可用
from trainer import Trainer, TrainerArgs
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainerConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer
from TTS.tts.models.xtts import Xtts
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from dataclasses import field, dataclass

# 自动接受条款
os.environ['COQUI_TOS_AGREED'] = '1'

# 设置设备
torch.set_num_threads(16)
device = torch.device("cpu")  # 强制使用CPU训练

# 设置批量大小和梯度累积步数
BATCH_SIZE = 4  # 增加批量大小以利用GPU并行处理能力
GRAD_ACUMM_STEPS = 8  # 调整梯度累积步数

# 数据集配置
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1",  # 修改为你的数据集路径
)

# 手动指定本地模型路径
model_path = "E:/大三下机器学习/pythonProject1/output/xtts_train/XTTS_v2.0_original_model_files"  # 修改为你的本地模型路径
config_path = os.path.join(model_path, "config.json")
vocab_path = os.path.join(model_path, "vocab.json")

# 创建一个新的配置类来使用ExtendedXttsAudioConfig
@dataclass
class ExtendedXttsAudioConfig(XttsAudioConfig):
    # 扩展XttsAudioConfig以解决PyTorch安全限制
    dvae_sample_rate: int = 22050

# 创建一个新的配置类来使用ExtendedXttsAudioConfig
class CustomGPTTrainerConfig(GPTTrainerConfig):
    audio: ExtendedXttsAudioConfig = field(default_factory=lambda: ExtendedXttsAudioConfig())

# 明确指定使用gpt_trainer中的扩展版本类
# 同时添加原始的XttsAudioConfig和XttsArgs类以解决序列化问题
torch.serialization.add_safe_globals([CustomGPTTrainerConfig, ExtendedXttsAudioConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, GPTArgs, XttsConfig])

# 设置当前文件路径
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# ========== 配置参数 ==========

# 输出路径：保存模型、配置和日志
OUT_PATH = os.path.join(CURRENT_PATH, "output/xtts_train")

# 数据集配置
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # 数据集格式
    dataset_name="ljspeech",  # 数据集名称
    path="./datasets/LJSpeech-1.1",  # 数据集路径
    meta_file_train="train.csv",  # 训练元数据文件
    meta_file_val="val.csv",  # 验证元数据文件
    language="en"  # 数据集语言
)

# 设置Torch线程数
torch.set_num_threads(16)

# 训练参数
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # 多GPU训练时设为False
START_WITH_EVAL = False  # 是否从评估开始
BATCH_SIZE = 1  # 减少批处理大小以降低内存需求
GRAD_ACUMM_STEPS = 1  # 减少梯度累积步数
# 建议：BATCH_SIZE * GRAD_ACUMM_STEPS 至少为252以提高训练效率

# 使用自定义配置类而不是XttsConfig
config = CustomGPTTrainerConfig(
    audio=ExtendedXttsAudioConfig()
)

# 更新配置参数
config.train_datasets = [dataset_config]
config.batch_size = BATCH_SIZE
config.grad_accum_steps = GRAD_ACUMM_STEPS
config.num_workers = 0  # 不使用多进程加载数据
config.num_eval_samples = 1
config.output_path = OUT_PATH
config.eval_split_max_size = 2  # 减少评估集大小以节省内存
config.print_step = 10  # 更频繁地打印训练进度
config.save_step = 50  # 更频繁地保存检查点
config.run_eval = False  # 暂时禁用评估以减少内存需求
config.mixed_precision = False  # 暂时禁用混合精度训练以避免梯度计算错误
config.precision = "fp32"  # 使用fp32精度
config.optimizer = "RAdam"  # 添加优化器类型
config.optimizer_params = {"weight_decay": 1e-4}  # 添加优化器参数

# 确保使用GPTArgs并设置必要的参数
# 设置必要的参数
gpt_args = GPTArgs()
gpt_args.tokenizer_file = vocab_path  # 使用预训练模型的词汇表
gpt_args.dvae_checkpoint = os.path.join(model_path, "dvae.pth")
gpt_args.gpt_checkpoint = os.path.join(model_path, "model.pth")
# 使用本地的mel_stats.pth文件而不是默认的URL
gpt_args.mel_norm_file = os.path.join(model_path, "mel_stats.pth")
# 设置正确的gpt_num_audio_tokens值以匹配预训练模型
# 不再手动设置tokenizer_file，让模型使用预训练的词汇表
# gpt_args.tokenizer_file = os.path.join(model_path, "vocab.json")  # 注释掉这一行

# 使用预训练模型的配置
gpt_args.gpt_num_audio_tokens = 1026  # XTTS v2.0 uses 1026 audio tokens
gpt_args.gpt_start_audio_token = 1024  # Start token index
gpt_args.gpt_stop_audio_token = 1025  # Stop token index
# 设置perceiver resampler参数以匹配训练时的配置
gpt_args.gpt_use_perceiver_resampler = False  # 确保与训练时保持一致
# 移除xtts_checkpoint属性，因为它不存在于GPTArgs中
if hasattr(gpt_args, 'xtts_checkpoint'):
    delattr(gpt_args, 'xtts_checkpoint')

# 将GPTArgs设置为config.model_args
config.model_args = gpt_args

config.test_sentences = [
    {
        "text": "This is a test sentence.",
        "speaker": "ljspeech",
        "language": "en"
    }
]

# 加载数据集
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=None,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1  # 使用10%的数据作为验证集
)

# 初始化模型
model = GPTTrainer.init_from_config(config)  # 使用GPTTrainer而不是Xtts

# 初始化训练器
trainer = Trainer(
    TrainerArgs(),  # 移除所有参数，使用默认初始化
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# 开始训练
trainer.fit()

# 保存最终模型
trainer.save_best_model()
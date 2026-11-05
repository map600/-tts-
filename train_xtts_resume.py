#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XTTS 微调续训脚本（GPU）

从 checkpoint_11950.pth 恢复训练（包含优化器状态，无缝续训）。
- 设备：GPU（CUDA 可用时自动使用）
- 架构：保持 gpt_use_perceiver_resampler=False，与现有微调权重一致
- 训练量：默认再训 5 个 epoch（每 epoch ≈ 12,000 步 ≈ 2.3 小时 @RTX4060）
- 显存：batch=1 + fp32（8GB 显存验证过可跑）；grad_accum_steps=32 提升梯度稳定性（不增加显存）

运行：python train_xtts_resume.py
"""

import bootstrap_fake_librosa  # noqa: F401  必须最先导入：绕过本机 librosa 导入卡死问题

import os
import time

# 禁用 coqui 的联网统计 ping（本机 SSL 证书校验失败会崩溃，且与训练无关；
# 必须在 import trainer 之前设置，因为该变量在 trainer.analytics 导入时读取）
os.environ['TRAINER_TELEMETRY'] = '0'

# 减少 CUDA 显存碎片（CUDA OOM 时 PyTorch 官方建议；必须在 import torch 之前设置）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

from dataclasses import field, dataclass

from trainer import Trainer, TrainerArgs

# 双保险：无论环境变量是否生效，直接把 ping 函数替换为空操作
import trainer.analytics as _analytics
_analytics.ping_training_run = lambda: None
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

os.environ['COQUI_TOS_AGREED'] = '1'

# ---- 与 checkpoint 内嵌配置一致的桩类（pickle 反序列化需要） ----
@dataclass
class ExtendedXttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050


class CustomGPTTrainerConfig(GPTTrainerConfig):
    audio: ExtendedXttsAudioConfig = field(default_factory=lambda: ExtendedXttsAudioConfig())


torch.serialization.add_safe_globals(
    [CustomGPTTrainerConfig, ExtendedXttsAudioConfig, XttsConfig, XttsAudioConfig,
     XttsArgs, BaseDatasetConfig, GPTArgs]
)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(CURRENT_PATH, "output/xtts_train_resume")
MODEL_FILES = os.path.join(CURRENT_PATH, "output/xtts_train/XTTS_v2.0_original_model_files")
CKPT = os.path.join(
    CURRENT_PATH,
    "output/xtts_train/run-December-16-2025_12+41AM-0000000/checkpoint_11950.pth",
)

# ========== 可调参数 ==========
EPOCHS = 5            # 续训轮数（每轮 = 全量 12,000 条数据）
BATCH_SIZE = 1        # 8GB 显存上限，保持 1
GRAD_ACUMM_STEPS = 1  # 必须为 1：本框架 XTTS 是多优化器训练，不支持梯度累积（≠1 会在开训时报 ValueError）
LR = 5e-06            # 与之前训练一致
SAVE_STEP = 500       # 每 500 步存一次 checkpoint（每个约 5GB，别存太频繁）
SAVE_N_CHECKPOINTS = 2  # 只保留最近 2 个

# ========== 数据集配置 ==========
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path=os.path.join(CURRENT_PATH, "datasets/LJSpeech-1.1"),  # 绝对路径，任何目录启动都可用
    meta_file_train="train.csv",
    meta_file_val="val.csv",
    language="en",
)

# ========== 训练配置 ==========
torch.set_num_threads(16)

config = CustomGPTTrainerConfig(audio=ExtendedXttsAudioConfig())
config.train_datasets = [dataset_config]
config.batch_size = BATCH_SIZE
config.epochs = EPOCHS
config.num_workers = 0
config.output_path = OUT_PATH
config.eval_split_max_size = 2
config.print_step = 50
config.save_step = SAVE_STEP
config.save_n_checkpoints = SAVE_N_CHECKPOINTS
config.run_eval = False
config.mixed_precision = False   # 与之前一致（fp32）
config.precision = "fp32"
config.optimizer = "RAdam"
config.optimizer_params = {"weight_decay": 1e-4}
config.lr = LR

# 唯一运行名（时间+进程号）：防止同一分钟多次启动导致运行目录同名、日志文件互相锁死
config.run_name = "xtts_resume_%s_%d" % (time.strftime("%m%d_%H%M%S"), os.getpid())

# 模型参数（与之前训练完全一致，perceiver=False 必须保持）
gpt_args = GPTArgs()
gpt_args.tokenizer_file = os.path.join(MODEL_FILES, "vocab.json")
gpt_args.dvae_checkpoint = os.path.join(MODEL_FILES, "dvae.pth")
gpt_args.gpt_checkpoint = os.path.join(MODEL_FILES, "model.pth")
gpt_args.mel_norm_file = os.path.join(MODEL_FILES, "mel_stats.pth")
gpt_args.gpt_num_audio_tokens = 1026
gpt_args.gpt_start_audio_token = 1024
gpt_args.gpt_stop_audio_token = 1025
gpt_args.gpt_use_perceiver_resampler = False  # 与 checkpoint 内嵌配置一致，勿改
if hasattr(gpt_args, "xtts_checkpoint"):
    delattr(gpt_args, "xtts_checkpoint")
config.model_args = gpt_args

config.test_sentences = [
    {"text": "This is a test sentence.", "speaker": "ljspeech", "language": "en"}
]

# ========== 加载数据 & 开始训练 ==========
print("Loading dataset...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=None,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1,
)
print(f"train: {len(train_samples)} samples, eval: {len(eval_samples)} samples")

model = GPTTrainer.init_from_config(config)

# 恢复训练：restore_path 和 grad_accum_steps 都放在 TrainerArgs（本版本 GPTTrainerConfig 无这些字段）
trainer = Trainer(
    TrainerArgs(restore_path=CKPT, grad_accum_steps=GRAD_ACUMM_STEPS),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# 省显存：hifigan 解码器在微调中冻结（只有 GPT 参与训练）且训练前向不使用，
# 挪到 CPU（约 104MB）。已确认 trainer 的 fit() 不会再次调用 model.cuda()，此操作安全。
if hasattr(trainer.model, "xtts") and hasattr(trainer.model.xtts, "hifigan_decoder"):
    trainer.model.xtts.hifigan_decoder.cpu()
    torch.cuda.empty_cache()
    print("[VRAM] hifigan_decoder moved to CPU (~104 MB saved)")

trainer.fit()

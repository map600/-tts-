import os
import torch
# 设置环境变量以避免Git分支问题
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, ExtendedXttsAudioConfig
from TTS.utils.manage import ModelManager

# 添加对XTTS配置类的信任以解决PyTorch安全限制
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from dataclasses import field
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

# 创建一个新的配置类来使用ExtendedXttsAudioConfig
class CustomGPTTrainerConfig(GPTTrainerConfig):
    audio: ExtendedXttsAudioConfig = field(default_factory=ExtendedXttsAudioConfig)

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

// 设置Torch线程数
torch.set_num_threads(16)

# 训练参数
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # 多GPU训练时设为False
START_WITH_EVAL = False  # 是否从评估开始
BATCH_SIZE = 6  # 增加批处理大小从3到6
GRAD_ACUMM_STEPS = 42  # 调整梯度累积步数以保持相似的effective batch size
// 建议：BATCH_SIZE * GRAD_ACUMM_STEPS 至少为252以提高训练效率

# ========== 下载预训练模型 ==========

# 定义预训练模型下载路径
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE模型文件（用于音频编码）
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# 设置下载文件的本地路径
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# 如果文件不存在则下载
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > 正在下载DVAE模型文件...")
    manager = ModelManager()
    manager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# XTTS预训练模型文件
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# 设置下载文件的本地路径
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

# 如果文件不存在则下载
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > 正在下载XTTS v2.0模型文件...")
    manager = ModelManager()
    manager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# ========== 模型和训练配置 ==========

# 初始化GPT模型参数 (使用XTTS v2配置)
model_args = GPTArgs(
    max_conditioning_length=132300,  # 最大条件音频长度（6秒）
    min_conditioning_length=66150,    # 最小条件音频长度（3秒）
    debug_loading_failures=False,
    max_wav_length=255995,            # 最大音频长度（约11.6秒）
    max_text_length=200,              # 最大文本长度
    mel_norm_file=MEL_NORM_FILE,      # 梅尔频谱归一化文件
    dvae_checkpoint=DVAE_CHECKPOINT,  # DVAE模型检查点
    xtts_checkpoint=XTTS_CHECKPOINT,  # XTTS模型检查点
    tokenizer_file=TOKENIZER_FILE,    # 分词器文件
    gpt_num_audio_tokens=1026,        # XTTS v2使用的参数
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

# 初始化音频配置
audio_config = ExtendedXttsAudioConfig(
    sample_rate=22050,            # GPT模型的采样率
    dvae_sample_rate=22050,       # DVAE模型的采样率
    output_sample_rate=24000,     # 输出音频的采样率
)

# 初始化训练配置
config = CustomGPTTrainerConfig(
    output_path=OUT_PATH,                 # 输出路径
    model_args=model_args,                # 模型参数
    run_name="XTTS_finetune",
    project_name="XTTS_trainer",
    run_description="XTTS fine-tuning",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,                   # 音频配置
    batch_size=BATCH_SIZE,                # 批量大小
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,           # 评估批量大小
    num_loader_workers=8,                 # 数据加载线程数
    eval_split_max_size=100,              # 评估集最大样本数
    print_step=50,                        # 打印间隔步数
    plot_step=100,                        # 绘图间隔步数
    log_model_step=1000,                  # 模型日志间隔步数
    save_step=10000,                      # 保存模型间隔步数
    save_n_checkpoints=2,                 # 保存的检查点数量
    save_checkpoints=True,                # 是否保存检查点
    print_eval=False,
    optimizer="AdamW",                    # 优化器
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,                             # 学习率
    lr_scheduler="MultiStepLR",           # 学习率调度器
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,  # 仅对权重应用权重衰减
    test_sentences=[],                    # 测试句子
)

# ========== 加载数据集 ==========

print(" > 正在加载数据集...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,                          # 数据集配置
    eval_split=True,                         # 是否划分评估集
    eval_split_max_size=config.eval_split_max_size,  # 评估集最大样本数
    eval_split_size=config.eval_split_size,  # 评估集比例
)

# ========== 初始化训练器 ==========

print(" > 正在初始化训练器...")
model = GPTTrainer.init_from_config(config)

# 添加错误处理以绕过Git分支获取问题
import trainer.generic_utils
original_get_git_branch = trainer.generic_utils.get_git_branch

def safe_get_git_branch():
    try:
        return original_get_git_branch()
    except Exception:
        return "unknown"

# 直接替换get_git_branch函数以避免StopIteration异常
def patched_get_git_branch():
    import subprocess
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        # 使用生成器表达式并提供默认值以避免StopIteration
        current = next((line for line in out.split("\n") if line.startswith("*")), "")
        if current:
            return current.replace("* ", "")
        else:
            return "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

trainer.generic_utils.get_git_branch = patched_get_git_branch

trainer = Trainer(
    TrainerArgs(
        restore_path=None,                          // 恢复路径（不使用，通过xtts_checkpoint参数恢复）
        skip_train_epoch=False,                     // 是否跳过训练epoch
        start_with_eval=START_WITH_EVAL,            // 是否从评估开始
        grad_accum_steps=GRAD_ACUMM_STEPS,          // 梯度累积步数
        mixed_precision=True,                       // 启用混合精度训练
        precision="fp16"                           // 使用fp16精度
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# ========== 开始训练 ==========

print(" > 开始训练...")
trainer.fit()
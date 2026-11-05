# XTTS模型训练指南

## 1. 环境准备

### 1.1 安装依赖

首先，确保你已经安装了Python 3.8+。然后，安装TTS库和其他依赖：

```bash
# 安装TTS库
pip install -e ./TTS-dev

# 安装训练所需的额外依赖
pip install -r ./TTS-dev/requirements.txt
pip install -r ./TTS-dev/TTS/demos/xtts_ft_demo/requirements.txt

# 如果你计划使用GPU加速，确保安装了正确版本的PyTorch
# 例如，对于CUDA 11.8：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1.2 检查环境

运行以下命令检查TTS是否正确安装：

```bash
tts --list_models
```

## 2. 数据集准备

### 2.1 支持的数据集格式

XTTS支持多种数据集格式，包括：
- LJSpeech (单说话人英文)
- VCTK (多说话人英文)
- LibriTTS (多说话人英文)
- Common Voice (多语言)

### 2.2 数据集结构

以LJSpeech为例，数据集结构应该是：

```
ljspeech/
├── wavs/
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   └── ...
└── metadata.csv
```

metadata.csv的格式：
```
LJ001-0001|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition
```

### 2.3 准备自己的数据集

1. 收集音频文件（建议采样率22050Hz或24000Hz）
2. 创建metadata.csv文件，格式如上
3. 确保音频文件和文本对应正确

## 3. 训练脚本配置

### 3.1 主要配置参数

打开`train_xtts.py`文件，修改以下关键参数：

```python
# 数据集配置
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # 数据集格式
    dataset_name="ljspeech",  # 数据集名称
    path="path/to/your/dataset",  # 你的数据集路径
    meta_file_train="metadata.csv",  # 训练元数据文件
    language="en"  # 数据集语言
)
```

### 3.2 训练参数调整

根据你的硬件配置调整以下参数：

```python
# 训练参数
BATCH_SIZE = 3  # 批量大小
GRAD_ACUMM_STEPS = 84  # 梯度累积步数
# 建议：BATCH_SIZE * GRAD_ACUMM_STEPS 至少为252以提高训练效率
```

## 4. 运行训练

### 4.1 开始训练

运行以下命令开始训练：

```bash
python train_xtts.py
```

### 4.2 训练过程

训练过程中，脚本会：
1. 下载必要的预训练模型文件
2. 加载和预处理数据集
3. 初始化模型和训练器
4. 开始训练循环
5. 定期保存模型检查点
6. 进行评估

### 4.3 监控训练

使用TensorBoard监控训练进度：

```bash
tensorboard --logdir output/xtts_train/logs
```

## 5. 模型评估

### 5.1 评估指标

训练过程中，模型会自动评估以下指标：
- 损失值（Loss）
- 文本和音频的交叉熵（CE）
- 生成音频质量

### 5.2 手动评估

训练完成后，你可以使用以下代码测试模型：

```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# 加载配置和模型
config = XttsConfig()
config.load_json("output/xtts_train/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="output/xtts_train/", use_deepspeed=False)
model.cuda()

# 计算说话人特征
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["path/to/reference.wav"])

# 生成语音
out = model.inference(
    "Hello, this is a test of the XTTS model.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7,
)

# 保存生成的音频
torchaudio.save("test_output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
```

## 6. 模型微调

### 6.1 微调现有模型

如果你想微调现有模型，修改`model_args`中的`xtts_checkpoint`参数：

```python
model_args = GPTArgs(
    # ... 其他参数 ...
    xtts_checkpoint="path/to/existing/model.pth",  # 现有模型检查点
    # ... 其他参数 ...
)
```

### 6.2 调整学习率

微调时，建议使用较小的学习率：

```python
config = GPTTrainerConfig(
    # ... 其他参数 ...
    lr=1e-06,  # 微调时使用较小的学习率
    # ... 其他参数 ...
)
```

## 7. 常见问题

### 7.1 内存不足

- 减少`BATCH_SIZE`
- 增加`GRAD_ACUMM_STEPS`
- 减小`max_wav_length`参数

### 7.2 训练速度慢

- 确保启用了混合精度训练（`use_amp=True`）
- 增加`num_loader_workers`
- 确保`BATCH_SIZE * GRAD_ACUMM_STEPS`至少为252

### 7.3 生成音频质量差

- 增加训练轮次
- 确保数据集质量高（清晰的音频、准确的文本对应）
- 调整生成参数（温度、top_k、top_p等）

## 8. 高级功能

### 8.1 多GPU训练

设置`use_ddp=True`并调整其他参数：

```python
config = GPTTrainerConfig(
    # ... 其他参数 ...
    use_ddp=True,  # 启用分布式数据并行
    optimizer_wd_only_on_weights=False,  # 多GPU时设为False
    # ... 其他参数 ...
)
```

### 8.2 语音克隆

XTTS支持语音克隆功能，你可以使用少量参考音频生成特定说话人的语音：

```python
# 计算特定说话人的特征
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["path/to/speaker/reference.wav"])

# 使用该说话人特征生成语音
out = model.inference(
    "Text to synthesize",
    "en",
    gpt_cond_latent,
    speaker_embedding,
)
```

## 9. 参考资源

- [XTTS官方文档](https://tts.readthedocs.io/en/dev/models/xtts.html)
- [Coqui TTS GitHub仓库](https://github.com/coqui-ai/TTS)
- [XTTS训练教程](https://colab.research.google.com/drive/1GiI4_X724M8q2W-zZ-jXo7cWTV7RfaH-?usp=sharing)

## 10. 许可证

XTTS模型使用[Coqui Public Model License](https://coqui.ai/cpml)。

---

如果有任何问题，请参考官方文档或加入Coqui社区寻求帮助：
- [Discord](https://discord.gg/fBC58unbKE)
- [Twitter](https://twitter.com/coqui_ai)
- [GitHub Issues](https://github.com/coqui-ai/TTS/issues)
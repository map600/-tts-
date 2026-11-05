# XTTS 语音合成项目

基于 **Coqui XTTS v2.0** 的语音合成与语音克隆项目：支持 **中/英等 17 种语言** 的文本转语音，通过一段参考音频即可克隆任意音色。项目包含数据集准备、模型微调训练、权重提取、桌面 GUI 与 **Web 界面** 的完整链路。

> 模型权重基于官方 XTTS v2.0 预训练模型，并在 LJSpeech（英文单说话人）数据集上微调了 **1 个 epoch（12,000 步）**。
> XTTS 模型权重遵循 [Coqui Public Model License](https://coqui.ai/cpml)。

---

## 功能特性

- 🌐 **Web 界面**：浏览器中完成文本合成、参考音色上传、参数调节、在线播放与下载
- 🖥️ **桌面 GUI**：tkinter 界面（文本/批量/设置三个选项卡，或简化单页版）
- 🗣️ **语音克隆**：一段 3~10 秒参考音频即可复刻音色
- 🌍 **多语言**：zh-cn / en / es / fr / de / it / pt / pl / tr / ru / nl / cs / ar / hu / ko / ja / hi
- 🎛️ **可调参数**：temperature、top-k、top-p、重复惩罚、长度惩罚
- 🚀 **GPU 加速**：自动检测 CUDA（无 GPU 时回退 CPU）
- ✅ **权重验证工具**：张量级对比，确认微调权重已真正生效

## 目录结构

```
pythonProject1/
├── web_tts_server.py          # Web 服务（FastAPI + 推理后端）
├── web_ui/
│   ├── index.html             # Web 前端页面（自包含，无需构建）
│   └── samples/sample_ref.wav # 内置示例音色
├── run_web_ui.bat             # 一键启动 Web 界面
├── xtts_gui.py                # 桌面 GUI（功能完整版）
├── real_xtts_gui.py           # 桌面 GUI（简洁版）
├── run_xtts_gui.bat           # 一键启动桌面 GUI
├── train_xtts.py              # 微调训练脚本（从预训练开始）
├── train_xtts_resume.py       # 续训脚本（从 checkpoint 恢复）
├── extract_model_mmap.py      # 从训练 checkpoint 提取纯权重
├── verify_model_weights.py    # 权重生效验证（张量级对比）
├── batch_synthesis.py         # 命令行批量合成
├── test_chinese_synthesis.py  # 中文合成测试
├── test_finetuned_model.py    # 微调模型推理测试
├── bootstrap_fake_librosa.py  # 兼容层：绕过部分机器上 librosa 导入卡死问题
├── final_model/               # 推理用模型文件（约 2.2GB，不入库）
│   ├── model.pth              #   GPT 权重（微调后）
│   ├── dvae.pth               #   DVAE 音频编解码器
│   ├── vocab.json / config.json / complete_config.json / mel_stats.pth
├── datasets/LJSpeech-1.1/     # 训练数据（不入库）
├── output/xtts_train/         # 训练运行记录与 checkpoint（不入库）
└── TTS-dev/                   # Coqui TTS 本地源码副本（依赖项）
```

## 环境安装

```bash
# 1. 创建环境（Python 3.10）
conda create -n myenv python=3.10
conda activate myenv

# 2. 安装 PyTorch（GPU 用户按 https://pytorch.org 选择对应 CUDA 版本）
#    CPU 示例：pip install torch torchaudio

# 3. 安装其余依赖
pip install -r requirements.txt
```

## 模型文件准备

模型权重共约 **2.2GB**，未包含在本仓库中（超过 GitHub 限制）。请将以下文件放入 `final_model/` 目录：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.pth` | ~1.9GB | GPT 权重（本项目微调后版本，或官方预训练版） |
| `dvae.pth` | ~210MB | DVAE 音频编解码器 |
| `vocab.json` | ~360KB | 词表 |
| `config.json` | 小 | 最小配置 |
| `complete_config.json` | 小 | 完整推理配置 |
| `mel_stats.pth` | 小 | 梅尔谱归一化统计 |

获取方式：官方预训练文件见 Coqui XTTS-v2 发布页；本项目的微调权重可联系仓库所有者获取（或自行按下方"训练"一节产出）。

## 使用方式

### 1. Web 界面（推荐）

```bash
python web_tts_server.py     # Windows 可直接双击 run_web_ui.bat
```

浏览器打开 **http://127.0.0.1:8000** ：输入文本 → 上传参考音频（或勾选内置示例音色）→ 选择语言 → 点"开始合成"→ 在线试听/下载。

### 2. 桌面 GUI

```bash
python xtts_gui.py           # 或双击 run_xtts_gui.bat
```

### 3. 命令行

```bash
python test_finetuned_model.py    # 英文合成测试
python test_chinese_synthesis.py  # 中文合成测试（注意修改参考音频路径）
python batch_synthesis.py         # 批量合成（脚本内编辑文本列表）
```

### 4. 权重生效验证

```bash
python verify_model_weights.py
```

对比 `final_model/model.pth`、训练 checkpoint 与官方预训练权重，输出张量级对比结果。

## 训练与续训

1. **数据准备**：下载 LJSpeech 数据集并切分
   ```bash
   python download_ljspeech.py   # 下载+解压+自动切分 train/val
   # 或手动放置后用： python split_dataset.py
   ```
2. **首次微调**：`python train_xtts.py`（按需修改其中路径与超参数）
3. **续训**：`python train_xtts_resume.py`（从指定 checkpoint 恢复模型+优化器，无缝续训）
4. **提取权重**：训练完成后，修改 `extract_model_mmap.py` 中的 `CKPT` 路径为最新 checkpoint，运行后权重写入 `final_model/model.pth`

**训练注意事项**（本机验证经验）：

- 需 8GB 以上显存（RTX 4060 8GB 可跑 batch=1 + fp32；若 OOM：关闭占显存的程序、脚本已内置 `expandable_segments` 与冻结模块下放 CPU 的优化）
- XTTS 在本框架中为多优化器训练，**不支持梯度累积**（`grad_accum_steps` 必须为 1）
- 微调架构使用 `gpt_use_perceiver_resampler=False`（conditioning_encoder），推理脚本已同步保持
- 若出现 librosa 导入长时间卡死，确认脚本顶部已导入 `bootstrap_fake_librosa`

## 常见问题

| 现象 | 处理 |
|---|---|
| librosa 导入卡死 | 脚本已内置 `bootstrap_fake_librosa` 兼容层，确保它在文件最顶部导入 |
| 训练开训即 CUDA OOM | 关闭浏览器等占显存程序；保持 batch=1；`max_wav_length` 可降到 10 秒（会跳过超长样本） |
| 合成时 `WinError 32` 锁文件 | 多为崩溃后的清理噪音；不要同一分钟重复启动训练 |
| 报 SSL 证书错误 | 已通过 `TRAINER_TELEMETRY=0` 禁用框架联网统计 |
| 音色与预训练差别不大 | 微调仅 1 个 epoch，变化本就不明显；续训 3~5 个 epoch 后效果更显著 |

## 许可证

- **模型权重**：XTTS v2.0 遵循 [Coqui Public Model License](https://coqui.ai/cpml)
- **LJSpeech 数据集**：公有领域（Public Domain）
- **本项目代码**：可自行选择许可（默认未声明；如需可添加 MIT License）

## 参考

- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [XTTS 官方文档](https://tts.readthedocs.io/en/dev/models/xtts.html)

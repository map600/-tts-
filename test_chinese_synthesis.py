#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bootstrap_fake_librosa  # noqa: F401  必须最先导入：绕过本机 librosa 导入卡死问题

import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# 添加必要的类到安全全局变量列表中以解决PyTorch的安全限制
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

def main():
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 设置模型路径
    model_path = "final_model"
    
    # 加载配置
    print("Loading model configuration...")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "complete_config.json"))
    
    # 微调权重使用 conditioning_encoder 架构（gpt_use_perceiver_resampler=False），
    # 必须保持 False，否则与 final_model/model.pth 的键结构不匹配
    if hasattr(config, 'model_args') and hasattr(config.model_args, 'gpt_use_perceiver_resampler'):
        config.model_args.gpt_use_perceiver_resampler = False
    
    # 初始化XTTS模型
    print("Initializing XTTS model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, 
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"), 
        use_deepspeed=False,
        eval=True
    )
    model.to(device)  # 将模型移动到设备
    
    # 计算说话人潜变量
    print("Computing speaker latents...")
    # 使用您指定的音频文件作为参考
    speaker_wav = "E:/大三下机器学习/音频转码_爱给网_aigei_com.wav"
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)
    
    # 测试中文文本
    text = "八九玄功，斗战神通。"
    
    # 生成中文语音
    print("Generating Chinese speech...")
    out = model.inference(
        text=text,
        language="zh",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True
    )
    
    # 保存输出
    output_path = "chinese_output_custom.wav"
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    
    print(f"Chinese speech generated and saved to {output_path}")

if __name__ == "__main__":
    main()
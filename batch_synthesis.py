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

def synthesize_text(model, config, text, speaker_wav, output_path, language="en"):
    """合成单个文本"""
    # 计算说话人潜变量
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)
    
    # 生成语音
    out = model.inference(
        text=text,
        language=language,
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
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"语音已保存到: {output_path}")

def main():
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置模型路径
    model_path = "E:/大三下机器学习/pythonProject1/final_model"
    
    # 加载配置
    print("加载模型配置...")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "complete_config.json"))
    
    # 微调权重使用 conditioning_encoder 架构（gpt_use_perceiver_resampler=False），
    # 必须保持 False，否则与 final_model/model.pth 的键结构不匹配
    if hasattr(config, 'model_args') and hasattr(config.model_args, 'gpt_use_perceiver_resampler'):
        config.model_args.gpt_use_perceiver_resampler = False
    
    # 初始化XTTS模型
    print("初始化XTTS模型...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, 
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"), 
        use_deepspeed=False,
        eval=True
    )
    model.to(device)
    
    # 设置参考音频
    speaker_wav = "E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1/wavs/LJ001-0001.wav"
    
    # 定义要合成的文本列表
    texts = [
        "This is the first sentence for synthesis.",
        "Here is another example of synthesized speech.",
        "You can add as many texts as you want to this list."
    ]
    
    # 批量合成
    for i, text in enumerate(texts):
        output_path = f"E:/大三下机器学习/pythonProject1/batch_output_{i+1}.wav"
        print(f"正在合成第 {i+1} 个文本: {text}")
        synthesize_text(model, config, text, speaker_wav, output_path, "en")
    
    print("所有文本合成完成!")

if __name__ == "__main__":
    main()
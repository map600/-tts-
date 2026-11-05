#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bootstrap_fake_librosa  # noqa: F401  必须最先导入：绕过本机 librosa 导入卡死问题

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import torchaudio
from TTS.api import TTS

# 真实XTTS GUI应用程序

class RealXTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XTTS 真实语音合成工具")
        self.root.geometry("600x550")
        
        # 初始化TTS模型
        self.tts = None
        self.model_loaded = False
        
        # 检查必要的文件是否存在
        self.model_path = "E:/大三下机器学习/pythonProject1/final_model"
        self.check_model_files()
        
        # 创建界面
        self.create_widgets()
        
        # 在后台线程中加载模型
        self.status_var.set("正在加载模型，请稍候...")
        thread = threading.Thread(target=self.load_model)
        thread.daemon = True
        thread.start()
        
    def check_model_files(self):
        """检查模型文件是否存在"""
        required_files = ["model.pth", "config.json", "vocab.json"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_path, file)):
                missing_files.append(file)
                
        if missing_files:
            messagebox.showwarning("警告", f"缺少以下模型文件: {', '.join(missing_files)}\n请确保模型文件完整后再运行程序。")
            
    def load_model(self):
        """在后台线程中加载XTTS模型"""
        try:
            # 添加安全全局变量
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
            from TTS.tts.datasets import BaseDatasetConfig
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
            
            # 加载微调后的模型（使用 GPU 加速）
            self.tts = TTS(model_path=self.model_path, config_path=os.path.join(self.model_path, "config.json"), gpu=True)
            self.model_loaded = True
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set("模型加载完成，可以开始合成"))
            
        except Exception as e:
            self.root.after(0, lambda: self.on_model_load_error(str(e)))
            
    def on_model_load_error(self, error_msg):
        """模型加载出错后的处理"""
        self.status_var.set("模型加载失败")
        messagebox.showerror("错误", f"模型加载失败:\n{error_msg}\n\n请检查模型文件是否完整。")
            
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="XTTS 真实语音合成工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 文本输入
        ttk.Label(main_frame, text="输入文本:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.text_input = scrolledtext.ScrolledText(main_frame, height=6, width=50)
        self.text_input.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.text_input.insert(tk.END, "你好，欢迎使用XTTS真实语音合成工具！")
        
        # 参考音频
        ttk.Label(main_frame, text="参考音频:").grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        self.ref_audio_var = tk.StringVar()
        self.ref_audio_var.set("E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1/wavs/LJ001-0001.wav")
        ttk.Entry(main_frame, textvariable=self.ref_audio_var, width=50).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_ref_audio).grid(row=4, column=2, padx=(5, 0), pady=5)
        
        # 输出文件
        ttk.Label(main_frame, text="输出文件:").grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        self.output_var = tk.StringVar()
        self.output_var.set("E:/大三下机器学习/pythonProject1/real_output.wav")
        ttk.Entry(main_frame, textvariable=self.output_var, width=50).grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_output).grid(row=6, column=2, padx=(5, 0), pady=5)
        
        # 语言选择
        ttk.Label(main_frame, text="语言:").grid(row=7, column=0, sticky=tk.W, pady=(10, 5))
        self.language_var = tk.StringVar()
        self.language_var.set("zh")
        lang_combo = ttk.Combobox(main_frame, textvariable=self.language_var, 
                                 values=["en", "zh", "es", "fr", "de"], width=10)
        lang_combo.grid(row=8, column=0, sticky=tk.W, pady=5)
        
        # 合成按钮
        self.synthesize_btn = ttk.Button(main_frame, text="开始合成", command=self.start_synthesis)
        self.synthesize_btn.grid(row=9, column=0, columnspan=3, pady=30)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("正在初始化...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=10, column=0, columnspan=3, pady=5)
        
        # 使用说明
        instruction_text = """
使用说明:
1. 在文本框中输入要合成的文本（支持中英文）
2. 选择参考音频文件（用于声音克隆）
3. 设置输出文件路径
4. 选择合适的语言（中文请选择zh）
5. 点击"开始合成"按钮

注意: 首次运行需要加载模型，可能需要几分钟时间。
        """
        instruction_label = ttk.Label(main_frame, text=instruction_text, font=("Arial", 9), foreground="gray")
        instruction_label.grid(row=11, column=0, columnspan=3, pady=(20, 0), sticky=(tk.W, tk.E))
        
    def browse_ref_audio(self):
        filename = filedialog.askopenfilename(
            title="选择参考音频文件",
            filetypes=[("音频文件", "*.wav *.mp3"), ("所有文件", "*.*")]
        )
        if filename:
            self.ref_audio_var.set(filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="保存输出文件",
            defaultextension=".wav",
            filetypes=[("WAV文件", "*.wav"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_var.set(filename)
            
    def start_synthesis(self):
        """开始合成过程"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍后再试")
            return
            
        text = self.text_input.get("1.0", tk.END).strip()
        if not text or text == "请输入您要合成的文本...":
            messagebox.showwarning("警告", "请输入要合成的文本")
            return
            
        ref_audio = self.ref_audio_var.get()
        if not os.path.exists(ref_audio):
            messagebox.showwarning("警告", "参考音频文件不存在，请选择有效的音频文件")
            return
            
        # 禁用按钮并更新状态
        self.synthesize_btn.config(state="disabled")
        self.status_var.set("正在合成语音，请稍候...")
        
        # 在新线程中执行合成
        thread = threading.Thread(target=self.synthesize_audio)
        thread.daemon = True
        thread.start()
        
    def synthesize_audio(self):
        """实际音频合成过程"""
        try:
            text = self.text_input.get("1.0", tk.END).strip()
            ref_audio = self.ref_audio_var.get()
            output_file = self.output_var.get()
            language = self.language_var.get()
            
            # 使用XTTS模型进行合成
            self.tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language=language,
                file_path=output_file
            )
            
            # 更新UI
            self.root.after(0, self.on_synthesis_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.on_synthesis_error(str(e)))
            
    def on_synthesis_complete(self):
        """合成完成后的处理"""
        output_file = self.output_var.get()
        self.status_var.set("合成完成!")
        self.synthesize_btn.config(state="normal")
        messagebox.showinfo("完成", f"语音合成已完成!\n文件已保存到:\n{output_file}")
        
    def on_synthesis_error(self, error_msg):
        """合成出错后的处理"""
        self.status_var.set("合成失败")
        self.synthesize_btn.config(state="normal")
        messagebox.showerror("错误", f"语音合成失败:\n{error_msg}")

def main():
    root = tk.Tk()
    app = RealXTTSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
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
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# 添加必要的类到安全全局变量列表中以解决PyTorch的安全限制
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

class XTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XTTS 语音合成工具")
        self.root.geometry("800x600")
        
        # 设置模型路径
        self.model_path = "E:/大三下机器学习/pythonProject1/final_model"
        
        # 初始化模型相关变量
        self.model = None
        self.config = None
        self.is_model_loaded = False
        
        # 创建界面
        self.create_widgets()
        
        # 在后台线程中加载模型
        self.load_model_thread()
        
    def create_widgets(self):
        # 创建选项卡控件
        tab_control = ttk.Notebook(self.root)
        
        # 文本合成选项卡
        self.text_tab = ttk.Frame(tab_control)
        tab_control.add(self.text_tab, text='文本合成')
        
        # 批量合成选项卡
        self.batch_tab = ttk.Frame(tab_control)
        tab_control.add(self.batch_tab, text='批量合成')
        
        # 设置选项卡
        self.settings_tab = ttk.Frame(tab_control)
        tab_control.add(self.settings_tab, text='设置')
        
        tab_control.pack(expand=1, fill="both")
        
        # 文本合成选项卡内容
        self.create_text_tab()
        
        # 批量合成选项卡内容
        self.create_batch_tab()
        
        # 设置选项卡内容
        self.create_settings_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_text_tab(self):
        # 输入框架
        input_frame = ttk.LabelFrame(self.text_tab, text="输入设置")
        input_frame.pack(fill="x", padx=10, pady=10)
        
        # 文本输入
        ttk.Label(input_frame, text="合成文本:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.text_input = scrolledtext.ScrolledText(input_frame, height=5, width=70)
        self.text_input.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.text_input.insert(tk.END, "请输入您要合成的文本...")
        
        # 参考音频选择
        ttk.Label(input_frame, text="参考音频:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.ref_audio_var = tk.StringVar()
        self.ref_audio_var.set("E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1/wavs/LJ001-0001.wav")
        self.ref_audio_entry = ttk.Entry(input_frame, textvariable=self.ref_audio_var, width=60)
        self.ref_audio_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(input_frame, text="浏览", command=self.browse_ref_audio)
        browse_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # 语言选择
        ttk.Label(input_frame, text="语言:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.language_var = tk.StringVar()
        self.language_var.set("en")
        language_combo = ttk.Combobox(input_frame, textvariable=self.language_var, 
                                     values=["en", "zh", "es", "fr", "de", "it", "pt", "pl", "ar", "ru", "nl", "cs", "tr"])
        language_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # 参数设置框架
        params_frame = ttk.LabelFrame(self.text_tab, text="合成参数")
        params_frame.pack(fill="x", padx=10, pady=10)
        
        # 温度参数
        ttk.Label(params_frame, text="温度 (Temperature):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.temperature_var = tk.DoubleVar()
        self.temperature_var.set(0.7)
        temp_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.temperature_var, orient="horizontal")
        temp_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.temp_value_label = ttk.Label(params_frame, text="0.7")
        self.temp_value_label.grid(row=0, column=2, padx=5, pady=5)
        temp_scale.configure(command=lambda x: self.temp_value_label.configure(text=f"{float(x):.1f}"))
        
        # 其他参数
        ttk.Label(params_frame, text="Top-K:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.topk_var = tk.IntVar()
        self.topk_var.set(50)
        topk_spin = ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.topk_var, width=10)
        topk_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(params_frame, text="Top-P:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.topp_var = tk.DoubleVar()
        self.topp_var.set(0.85)
        topp_spin = ttk.Spinbox(params_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.topp_var, width=10)
        topp_spin.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 输出设置框架
        output_frame = ttk.LabelFrame(self.text_tab, text="输出设置")
        output_frame.pack(fill="x", padx=10, pady=10)
        
        # 输出文件路径
        ttk.Label(output_frame, text="输出文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.output_var = tk.StringVar()
        self.output_var.set("E:/大三下机器学习/pythonProject1/gui_output.wav")
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=60)
        self.output_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        output_browse_btn = ttk.Button(output_frame, text="浏览", command=self.browse_output)
        output_browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 合成按钮
        self.synthesize_btn = ttk.Button(self.text_tab, text="开始合成", command=self.synthesize_single)
        self.synthesize_btn.pack(pady=20)
        
        # 配置列权重
        input_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(1, weight=1)
        
    def create_batch_tab(self):
        # 批量合成说明
        info_label = ttk.Label(self.batch_tab, 
                              text="批量合成: 请准备一个文本文件，每行一个句子进行合成",
                              font=("Arial", 10))
        info_label.pack(pady=10)
        
        # 输入框架
        input_frame = ttk.LabelFrame(self.batch_tab, text="批量输入设置")
        input_frame.pack(fill="x", padx=10, pady=10)
        
        # 文本文件选择
        ttk.Label(input_frame, text="文本文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.batch_text_var = tk.StringVar()
        self.batch_text_entry = ttk.Entry(input_frame, textvariable=self.batch_text_var, width=60)
        self.batch_text_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        text_browse_btn = ttk.Button(input_frame, text="浏览", command=self.browse_batch_text)
        text_browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 参考音频选择
        ttk.Label(input_frame, text="参考音频:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.batch_ref_audio_var = tk.StringVar()
        self.batch_ref_audio_var.set("E:/大三下机器学习/pythonProject1/datasets/LJSpeech-1.1/wavs/LJ001-0001.wav")
        self.batch_ref_audio_entry = ttk.Entry(input_frame, textvariable=self.batch_ref_audio_var, width=60)
        self.batch_ref_audio_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ref_browse_btn = ttk.Button(input_frame, text="浏览", command=lambda: self.browse_file(self.batch_ref_audio_var))
        ref_browse_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # 输出目录选择
        ttk.Label(input_frame, text="输出目录:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.batch_output_var = tk.StringVar()
        self.batch_output_var.set("E:/大三下机器学习/pythonProject1/")
        self.batch_output_entry = ttk.Entry(input_frame, textvariable=self.batch_output_var, width=60)
        self.batch_output_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        output_browse_btn = ttk.Button(input_frame, text="浏览", command=self.browse_batch_output)
        output_browse_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # 语言选择
        ttk.Label(input_frame, text="语言:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.batch_language_var = tk.StringVar()
        self.batch_language_var.set("en")
        language_combo = ttk.Combobox(input_frame, textvariable=self.batch_language_var, 
                                     values=["en", "zh", "es", "fr", "de", "it", "pt", "pl", "ar", "ru", "nl", "cs", "tr"])
        language_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # 批量合成按钮
        self.batch_synthesize_btn = ttk.Button(self.batch_tab, text="开始批量合成", command=self.synthesize_batch)
        self.batch_synthesize_btn.pack(pady=20)
        
        # 配置列权重
        input_frame.columnconfigure(1, weight=1)
        
    def create_settings_tab(self):
        # 模型信息
        model_frame = ttk.LabelFrame(self.settings_tab, text="模型信息")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_path_var = tk.StringVar()
        self.model_path_var.set(self.model_path)
        model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=60, state="readonly")
        model_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # 模型状态
        self.model_status_var = tk.StringVar()
        self.model_status_var.set("模型未加载")
        self.model_status_label = ttk.Label(model_frame, textvariable=self.model_status_var)
        self.model_status_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # 重新加载模型按钮
        reload_btn = ttk.Button(model_frame, text="重新加载模型", command=self.load_model_thread)
        reload_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 高级参数设置
        advanced_frame = ttk.LabelFrame(self.settings_tab, text="高级参数")
        advanced_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(advanced_frame, text="长度惩罚:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.length_penalty_var = tk.DoubleVar()
        self.length_penalty_var.set(1.0)
        length_penalty_spin = ttk.Spinbox(advanced_frame, from_=0.1, to=2.0, increment=0.1, 
                                         textvariable=self.length_penalty_var, width=10)
        length_penalty_spin.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(advanced_frame, text="重复惩罚:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.repetition_penalty_var = tk.DoubleVar()
        self.repetition_penalty_var.set(10.0)
        repetition_penalty_spin = ttk.Spinbox(advanced_frame, from_=1.0, to=20.0, increment=0.5, 
                                             textvariable=self.repetition_penalty_var, width=10)
        repetition_penalty_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # 配置列权重
        model_frame.columnconfigure(1, weight=1)
        
    def browse_ref_audio(self):
        filename = filedialog.askopenfilename(
            title="选择参考音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac"), ("所有文件", "*.*")]
        )
        if filename:
            self.ref_audio_var.set(filename)
            
    def browse_file(self, var):
        filename = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("所有文件", "*.*")]
        )
        if filename:
            var.set(filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="保存输出文件",
            defaultextension=".wav",
            filetypes=[("WAV文件", "*.wav"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_var.set(filename)
            
    def browse_batch_text(self):
        filename = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            self.batch_text_var.set(filename)
            
    def browse_batch_output(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.batch_output_var.set(directory)
            
    def load_model_thread(self):
        """在后台线程中加载模型"""
        self.model_status_var.set("正在加载模型...")
        self.status_var.set("正在加载模型，请稍候...")
        self.synthesize_btn.config(state="disabled")
        self.batch_synthesize_btn.config(state="disabled")
        
        thread = threading.Thread(target=self.load_model)
        thread.daemon = True
        thread.start()
        
    def load_model(self):
        """加载XTTS模型"""
        try:
            # 加载配置
            self.config = XttsConfig()
            self.config.load_json(os.path.join(self.model_path, "complete_config.json"))
            
            # 微调权重使用 conditioning_encoder 架构（gpt_use_perceiver_resampler=False），
            # 必须保持 False，否则与 final_model/model.pth 的键结构不匹配
            if hasattr(self.config, 'model_args') and hasattr(self.config.model_args, 'gpt_use_perceiver_resampler'):
                self.config.model_args.gpt_use_perceiver_resampler = False
            
            # 初始化XTTS模型
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(
                self.config, 
                checkpoint_path=os.path.join(self.model_path, "model.pth"),
                vocab_path=os.path.join(self.model_path, "vocab.json"), 
                use_deepspeed=False,
                eval=True
            )
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            
            self.is_model_loaded = True
            
            # 更新UI
            self.root.after(0, self.on_model_loaded)
            
        except Exception as e:
            self.root.after(0, lambda: self.on_model_load_error(str(e)))
            
    def on_model_loaded(self):
        """模型加载完成后的回调"""
        self.model_status_var.set("模型已加载")
        self.status_var.set("模型加载完成，可以开始合成")
        self.synthesize_btn.config(state="normal")
        self.batch_synthesize_btn.config(state="normal")
        messagebox.showinfo("提示", "模型加载完成，可以开始语音合成了！")
        
    def on_model_load_error(self, error_msg):
        """模型加载错误的回调"""
        self.model_status_var.set("模型加载失败")
        self.status_var.set("模型加载失败")
        self.synthesize_btn.config(state="disabled")
        self.batch_synthesize_btn.config(state="disabled")
        messagebox.showerror("错误", f"模型加载失败: {error_msg}")
        
    def synthesize_single(self):
        """单条文本合成"""
        if not self.is_model_loaded:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍后再试")
            return
            
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要合成的文本")
            return
            
        # 在后台线程中执行合成
        self.status_var.set("正在合成语音...")
        self.synthesize_btn.config(state="disabled")
        
        thread = threading.Thread(target=self._synthesize_single_worker, args=(text,))
        thread.daemon = True
        thread.start()
        
    def _synthesize_single_worker(self, text):
        """单条文本合成工作线程"""
        try:
            # 获取参数
            ref_audio = self.ref_audio_var.get()
            language = self.language_var.get()
            temperature = self.temperature_var.get()
            top_k = self.topk_var.get()
            top_p = self.topp_var.get()
            length_penalty = self.length_penalty_var.get()
            repetition_penalty = self.repetition_penalty_var.get()
            output_path = self.output_var.get()
            
            # 计算说话人潜变量
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=ref_audio)
            
            # 生成语音
            out = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                enable_text_splitting=True
            )
            
            # 保存输出
            torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            
            # 更新UI
            self.root.after(0, lambda: self.on_synthesis_complete(output_path))
            
        except Exception as e:
            self.root.after(0, lambda: self.on_synthesis_error(str(e)))
            
    def on_synthesis_complete(self, output_path):
        """合成完成后的回调"""
        self.status_var.set("合成完成")
        self.synthesize_btn.config(state="normal")
        messagebox.showinfo("完成", f"语音合成已完成，文件保存到:\n{output_path}")
        
    def on_synthesis_error(self, error_msg):
        """合成错误的回调"""
        self.status_var.set("合成失败")
        self.synthesize_btn.config(state="normal")
        messagebox.showerror("错误", f"语音合成失败: {error_msg}")
        
    def synthesize_batch(self):
        """批量文本合成"""
        if not self.is_model_loaded:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍后再试")
            return
            
        text_file = self.batch_text_var.get()
        if not os.path.exists(text_file):
            messagebox.showwarning("警告", "请选择有效的文本文件")
            return
            
        output_dir = self.batch_output_var.get()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "请选择有效的输出目录")
            return
            
        # 在后台线程中执行批量合成
        self.status_var.set("正在进行批量合成...")
        self.batch_synthesize_btn.config(state="disabled")
        
        thread = threading.Thread(target=self._synthesize_batch_worker)
        thread.daemon = True
        thread.start()
        
    def _synthesize_batch_worker(self):
        """批量文本合成工作线程"""
        try:
            # 获取参数
            text_file = self.batch_text_var.get()
            ref_audio = self.batch_ref_audio_var.get()
            output_dir = self.batch_output_var.get()
            language = self.batch_language_var.get()
            temperature = self.temperature_var.get()
            top_k = self.topk_var.get()
            top_p = self.topp_var.get()
            length_penalty = self.length_penalty_var.get()
            repetition_penalty = self.repetition_penalty_var.get()
            
            # 读取文本文件
            with open(text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                
            if not texts:
                raise ValueError("文本文件中没有有效内容")
                
            # 计算说话人潜变量
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=ref_audio)
            
            # 逐条合成
            success_count = 0
            for i, text in enumerate(texts):
                try:
                    # 生成语音
                    out = self.model.inference(
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                        top_k=top_k,
                        top_p=top_p,
                        enable_text_splitting=True
                    )
                    
                    # 保存输出
                    output_path = os.path.join(output_dir, f"batch_output_{i+1:03d}.wav")
                    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
                    success_count += 1
                    
                except Exception as e:
                    print(f"合成第{i+1}条文本时出错: {e}")
                    continue
                    
            # 更新UI
            self.root.after(0, lambda: self.on_batch_synthesis_complete(success_count, len(texts)))
            
        except Exception as e:
            self.root.after(0, lambda: self.on_batch_synthesis_error(str(e)))
            
    def on_batch_synthesis_complete(self, success_count, total_count):
        """批量合成完成后的回调"""
        self.status_var.set("批量合成完成")
        self.batch_synthesize_btn.config(state="normal")
        messagebox.showinfo("完成", f"批量合成已完成!\n成功: {success_count}/{total_count} 条")
        
    def on_batch_synthesis_error(self, error_msg):
        """批量合成错误的回调"""
        self.status_var.set("批量合成失败")
        self.batch_synthesize_btn.config(state="normal")
        messagebox.showerror("错误", f"批量合成失败: {error_msg}")

def main():
    root = tk.Tk()
    app = XTTSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
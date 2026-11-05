# -*- coding: utf-8 -*-
"""验证 final_model/model.pth 是否真的是微调权重（张量级对比，无需加载 TTS 库）

对比三份文件：
  A = checkpoint_11950.pth（训练器保存的微调快照，权威基准）
  B = final_model/model.pth（推理实际使用的文件）
  C = XTTS_v2.0_original_model_files/model.pth（官方预训练原版）

结论判据：
  [1] B 的键集合 == A 的键集合
  [2] B 与 A 抽样张量完全相等  -> 证明 B 确实是微调权重
  [3] B 的 GPT 权重与 C 不同（平均绝对差 > 0）-> 证明不是预训练原版
  [4] B 与 C 的 mel_stats 完全相同 -> 对照组（微调不改变梅尔归一化统计）
"""
import copyreg
import random

import numpy as np
import torch


def make_stub(name):
    cls = type(name, (), {"__init__": lambda self, *a, **k: None})
    cls.__module__ = "TTS_MISSING"
    cls.__qualname__ = name
    return cls


registered = []
for mod, names in {
    "TTS.tts.configs.xtts_config": ["XttsConfig"],
    "TTS.tts.models.xtts": ["XttsAudioConfig", "XttsArgs"],
    "TTS.config.shared_configs": ["BaseDatasetConfig"],
    "TTS.tts.layers.xtts.trainer.gpt_trainer": ["GPTArgs", "GPTTrainerConfig"],
    "__main__": ["CustomGPTTrainerConfig", "ExtendedXttsAudioConfig"],
}.items():
    for n in names:
        s = make_stub(n)
        s.__module__ = mod
        registered.append(s)
registered += [copyreg._reconstructor, torch._utils._rebuild_parameter_with_state,
               torch._utils._rebuild_parameter, random.Random, np.ndarray, np.dtype,
               np.random.mtrand.RandomState]
try:
    from numpy._core.multiarray import _reconstruct
    registered.append(_reconstruct)
except Exception:
    pass
try:
    from numpy.core.multiarray import _reconstruct
    registered.append(_reconstruct)
except Exception:
    pass
torch.serialization.add_safe_globals(registered)

BASE = r"E:\大三下机器学习\pythonProject1"
A_PATH = BASE + r"\output\xtts_train\run-December-16-2025_12+41AM-0000000\checkpoint_11950.pth"
B_PATH = BASE + r"\final_model\model.pth"
C_PATH = BASE + r"\output\xtts_train\XTTS_v2.0_original_model_files\model.pth"

print("loading (mmap)...", flush=True)
A = torch.load(A_PATH, map_location="cpu", weights_only=True, mmap=True)["model"]
B = torch.load(B_PATH, map_location="cpu", weights_only=True, mmap=True)["model"]
C = torch.load(C_PATH, map_location="cpu", weights_only=True, mmap=True)["model"]

print(f"A (checkpoint) keys: {len(A)}")
print(f"B (final_model) keys: {len(B)}")
print(f"C (pretrained)  keys: {len(C)}")

print("\n[1] B keys == A keys:", set(B.keys()) == set(A.keys()))

sample_keys = [k for k in A if k.startswith("xtts.gpt")][:6]
print("\n[2] B vs A sampled tensors:")
all_equal = True
for k in sample_keys:
    eq = torch.equal(A[k], B[k])
    all_equal = all_equal and eq
    print(f"    {k}: {'EQUAL' if eq else 'DIFFERENT'}")

print("\n[3] B vs C (pretrained), GPT weight differences:")
pairs = [("xtts.gpt.gpt.wte.weight", "gpt.gpt.wte.weight"),
         ("xtts.gpt.final_norm.weight", "gpt.final_norm.weight"),
         ("xtts.gpt.text_head.weight", "gpt.text_head.weight")]
for bk, ck in pairs:
    if bk in B and ck in C:
        d = (B[bk].float() - C[ck].float()).abs()
        print(f"    {bk}: mean|diff|={d.mean().item():.6e}  max|diff|={d.max().item():.6e}")
    else:
        print(f"    {bk} <-> {ck}: key missing in one file")

print("\n[4] B vs C mel_stats (control, should be identical):")
if "xtts.mel_stats" in B and "mel_stats" in C:
    print("    identical:", torch.equal(B["xtts.mel_stats"], C["mel_stats"]))

print("\n===== 结论 =====")
print("B 就是 checkpoint_11950 的微调权重:", all_equal and set(B.keys()) == set(A.keys()))

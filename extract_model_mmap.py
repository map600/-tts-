# -*- coding: utf-8 -*-
"""从 checkpoint_11950.pth 提取模型权重 → final_model/model.pth

策略（内存受限环境，可用RAM仅~2GB）：
  1. 用 torch.load(..., weights_only=True, mmap=True) 内存映射加载：
     只解析 0.3MB 的结构描述，5GB 张量数据以映射方式按需读取，不占内存。
  2. 不导入 TTS 库（其依赖链在低内存下会卡死）；
     为 pickle 中出现的配置类注册"桩类"（假类），反序列化照样能完成。
  3. 取出 ckpt["model"]（纯模型权重），保存为 {"model": ...} 格式，
     TTS 的 Xtts.load_checkpoint() 只读取该键，并自动兼容 "xtts." 前缀。
"""
import copyreg
import os
import random
import time

import numpy as np
import torch

T = lambda m: print(f"[{time.time():.0f}] {m}", flush=True)


def make_stub(name):
    """创建带正确 module/qualname 的桩类，让 torch 安全反序列化能匹配 pickle 里的类名。"""
    cls = type(name, (), {"__init__": lambda self, *a, **k: None,
                          "__repr__": lambda self: f"<stub {name}>"})
    cls.__module__ = "TTS_MISSING"
    cls.__qualname__ = name
    return cls


# pickle 中会出现的类（对应训练脚本 __main__ 里定义的类 + TTS 配置类）
stubs = {
    "TTS.tts.configs.xtts_config": ["XttsConfig"],
    "TTS.tts.models.xtts": ["XttsAudioConfig", "XttsArgs"],
    "TTS.config.shared_configs": ["BaseDatasetConfig"],
    "TTS.tts.layers.xtts.trainer.gpt_trainer": ["GPTArgs", "GPTTrainerConfig"],
    "__main__": ["CustomGPTTrainerConfig", "ExtendedXttsAudioConfig"],
}
registered = []
for mod, names in stubs.items():
    for n in names:
        s = make_stub(n)
        s.__module__ = mod
        registered.append(s)

# 反序列化还原函数
registered.append(copyreg._reconstructor)
registered.append(torch._utils._rebuild_parameter_with_state)
registered.append(torch._utils._rebuild_parameter)
registered.append(random.Random)
registered.append(np.ndarray)
registered.append(np.dtype)
registered.append(np.random.mtrand.RandomState)
try:
    from numpy.core.multiarray import _reconstruct
    registered.append(_reconstruct)
except Exception as e:
    T("warn: numpy.core.multiarray._reconstruct not found: " + str(e))
try:
    from numpy._core.multiarray import _reconstruct as _reconstruct2
    registered.append(_reconstruct2)
except Exception:
    pass

torch.serialization.add_safe_globals(registered)

CKPT = r"E:\大三下机器学习\pythonProject1\output\xtts_train\run-December-16-2025_12+41AM-0000000\checkpoint_11950.pth"
OUT = r"E:\大三下机器学习\pythonProject1\final_model\model.pth"

T("Loading checkpoint with mmap (memory-mapped, low RAM)...")
t0 = time.time()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
T(f"Loaded in {time.time() - t0:.1f}s. Top-level keys: {list(ckpt.keys())}")

# ---- 打印结构信息 ----
model = ckpt["model"]
assert isinstance(model, dict), f"unexpected model type: {type(model)}"
import collections
prefixes = collections.Counter(k.split(".")[0] for k in model)
T(f"model dict: {len(model)} entries; first-level prefixes: {dict(prefixes)}")
perc = [k for k in model if "perceiver" in k]
T(f"perceiver keys: {len(perc)}, e.g. {perc[:2]}")
xtts_pref = sum(1 for k in model if k.startswith("xtts."))
T(f"'xtts.' prefixed keys: {xtts_pref}")
mod_pref = sum(1 for k in model if k.startswith("module."))
T(f"'module.' prefixed keys: {mod_pref}")
cfg = ckpt.get("config")
if cfg is not None:
    d = getattr(cfg, "__dict__", {})
    T(f"config type: {type(cfg).__module__}.{type(cfg).__qualname__}")
    T(f"config keys sample: {list(d.keys())[:15]}")
    ma = d.get("model_args")
    if ma is not None:
        ma_d = getattr(ma, "__dict__", {})
        T(f"model_args perceiver flag = {ma_d.get('gpt_use_perceiver_resampler')}")
        T(f"model_args tokens = {ma_d.get('gpt_num_audio_tokens')} / {ma_d.get('gpt_start_audio_token')} / {ma_d.get('gpt_stop_audio_token')}")

# ---- 提取权重 ----
if mod_pref and mod_pref == len(model):
    T("Stripping 'module.' prefix...")
    model = {k[len("module."):] if k.startswith("module.") else k: v for k, v in model.items()}

total_bytes = sum(v.numel() * v.element_size() for v in model.values() if isinstance(v, torch.Tensor))
n_params = sum(v.numel() for v in model.values() if isinstance(v, torch.Tensor))
T(f"Extracted weights: {n_params:,} params, {total_bytes / 1e9:.2f} GB")

T(f"Saving to {OUT} ...")
t0 = time.time()
torch.save({"model": model}, OUT)
T(f"Saved in {time.time() - t0:.1f}s; file size {os.path.getsize(OUT) / 1e9:.2f} GB")

# ---- 验证：重新以 mmap 读取新文件，检查键结构 ----
T("Verifying extracted file...")
v = torch.load(OUT, map_location="cpu", weights_only=True, mmap=True)
T(f"verify: top keys {list(v.keys())}, model entries {len(v['model'])}")
sample = [k for k in v["model"].keys()][:3]
T(f"verify: sample keys {sample}")
first = next(iter(v["model"].values()))
if isinstance(first, torch.Tensor):
    T(f"verify: first tensor shape {tuple(first.shape)} dtype {first.dtype}")
T("ALL DONE.")

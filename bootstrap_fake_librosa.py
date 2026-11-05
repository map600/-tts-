# -*- coding: utf-8 -*-
"""librosa 导入卡死绕过模块。

本机（Windows + 32核 + librosa 0.10.0/scipy 1.11.4）在导入 librosa 深层模块
（librosa.util.decorators 及其后的导入链）时会病理性卡死，已用 importtime
定位确认，与内存无关。XTTS 推理本身不使用 librosa（梅尔谱用纯 torch 实现），
只有 TTS 库的导入链会 import librosa。

因此本模块在导入时向 sys.modules 注入"假 librosa"：
  - 所有 from librosa import xxx 得到可调用的占位对象；
  - 子模块 import 得到预先注册的假模块；
  - 假模块带字符串 __file__ 与合法的 __spec__，避免 inspect/importlib 崩溃。

用法：在任何会导入 TTS 的脚本【最顶部】加一行：
    import bootstrap_fake_librosa  # noqa: F401
"""
import sys
import types
import importlib.machinery


class _Anything:
    """可调用、可继续访问属性的万能占位对象"""

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, n):
        return _ANYTHING

    def __repr__(self):
        return "<fake_librosa_placeholder>"


_ANYTHING = _Anything()


def _make_fake(name):
    m = types.ModuleType(name)
    m.__file__ = "<fake:%s>" % name   # 必须是字符串，防止 inspect 模块崩溃
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)  # importlib.util.find_spec 需要
    m.__path__ = []
    m.__all__ = []

    def _getattr(n):
        if n in ("__path__", "__file__", "__name__", "__spec__", "__loader__", "__package__"):
            raise AttributeError(n)
        return _ANYTHING

    m.__getattr__ = _getattr
    sys.modules[name] = m
    return m


librosa = _make_fake("librosa")
librosa.__version__ = "0.10.0"

# 预先注册常见子模块，避免 "import librosa.core.xxx" 报 ModuleNotFoundError
for _sub in ["core", "feature", "util", "display", "effects", "filters",
             "sequence", "beat", "onset", "segment", "decompose", "_cache",
             "core.audio", "core.spectrum", "core.pitch", "core.convert",
             "core.fft", "core.constantq", "util.utils", "util.exceptions",
             "util.decorators", "feature.spectral", "feature.rhythm"]:
    _make_fake("librosa." + _sub)

# 常用顶层名字，直接给占位（可调用）
for _n in ["load", "stft", "istft", "magphase", "pyin", "melspectrogram",
           "get_window", "power_to_db", "db_to_power", "amplitude_to_db",
           "db_to_amplitude", "resample", "frames_to_time", "time_to_frames",
           "note_to_hz", "hz_to_note", "trim", "pcen", "griffinlim",
           "cqt", "feature", "display", "effects", "onset", "beat",
           "sequence", "decompose", "filters", "segment"]:
    setattr(librosa, _n, _ANYTHING)

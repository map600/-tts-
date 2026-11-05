#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XTTS 语音合成 Web 服务

启动: python web_tts_server.py  (或双击 run_web_ui.bat)
访问: http://127.0.0.1:8000

功能:
  - 输入文本 + 上传参考音频（或使用内置示例音色）→ 合成语音
  - 支持中/英等 17 种语言
  - 可调 temperature / top_k / top_p / 重复惩罚 / 长度惩罚
  - 自动使用 GPU（不可用时回退 CPU）
"""
import bootstrap_fake_librosa  # noqa: F401  必须最先导入：绕过本机 librosa 导入卡死问题

import io
import os
import shutil
import tempfile
import threading
import time

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "final_model")
UI_DIR = os.path.join(CURRENT_PATH, "web_ui")
SAMPLE_REF = os.path.join(UI_DIR, "samples", "sample_ref.wav")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="XTTS 语音合成")

model = None
model_config = None
model_lock = threading.Lock()
load_started = time.time()

FALLBACK_LANGS = ["zh-cn", "en", "es", "fr", "de", "it", "pt", "pl",
                  "tr", "ru", "nl", "cs", "ar", "hu", "ko", "ja", "hi"]


def _load_model():
    global model, model_config
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "complete_config.json"))
    # 微调权重为 conditioning_encoder 架构（perceiver=False），必须保持
    if hasattr(config, "model_args") and hasattr(config.model_args, "gpt_use_perceiver_resampler"):
        config.model_args.gpt_use_perceiver_resampler = False
    m = Xtts.init_from_config(config)
    m.load_checkpoint(
        config,
        checkpoint_path=os.path.join(MODEL_DIR, "model.pth"),
        vocab_path=os.path.join(MODEL_DIR, "vocab.json"),
        use_deepspeed=False,
        eval=True,
    )
    m.to(DEVICE)
    m.eval()
    model, model_config = m, config


def _load_model_thread():
    try:
        _load_model()
        print(f"[web] 模型加载完成，设备: {DEVICE}，耗时 {time.time() - load_started:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[web] 模型加载失败: {e}", flush=True)


threading.Thread(target=_load_model_thread, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/api/status")
def status():
    return {
        "loaded": model is not None,
        "device": DEVICE,
        "loading_seconds": round(time.time() - load_started, 1),
        "languages": list(model_config.languages) if model_config is not None else FALLBACK_LANGS,
        "sample_ref_exists": os.path.exists(SAMPLE_REF),
    }


@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("zh-cn"),
    temperature: float = Form(0.7),
    top_k: int = Form(50),
    top_p: float = Form(0.85),
    repetition_penalty: float = Form(10.0),
    length_penalty: float = Form(1.0),
    use_sample_ref: bool = Form(False),
    file: UploadFile = File(None),
):
    if model is None:
        return JSONResponse({"error": "模型尚未加载完成，请稍候再试"}, status_code=503)

    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "请输入要合成的文本"}, status_code=400)

    tmp_path = None
    try:
        if use_sample_ref and os.path.exists(SAMPLE_REF):
            ref_path = SAMPLE_REF
        elif file is not None and file.filename:
            suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            ref_path = tmp_path
        else:
            return JSONResponse({"error": "请上传参考音频，或勾选使用内置示例音色"}, status_code=400)

        with model_lock:
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=ref_path)
            out = model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                enable_text_splitting=True,
            )

        wav = out["wav"]
        buf = io.BytesIO()
        torchaudio.save(buf, torch.tensor(wav).unsqueeze(0), 24000, format="wav")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="tts_output.wav"'},
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"合成失败: {e}"}, status_code=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    print("[web] 模型正在后台加载（首次约 1-2 分钟）...", flush=True)
    print("[web] 服务地址: http://127.0.0.1:8000", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)

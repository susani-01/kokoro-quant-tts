import torch
import soundfile as sf
import numpy as np
import time
import os
import io
from kokoro import KPipeline
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

torch.backends.quantized.engine='qnnpack'

app = FastAPI(title="TTS Quantization Demo", version="1.0.0")

baseline_pipeline = KPipeline(lang_code='a')
quantized_pipeline = KPipeline(lang_code='a')
quantized_pipeline.model = torch.quantization.quantize_dynamic(
    quantized_pipeline.model,
    {torch.nn.Linear},
    dtype=torch.qint8,
    inplace=True
)

def run_tts(pipeline, text):
    start = time.time()
    generator = pipeline(text, voice='af_heart', speed=1.0)
    chunks = [audio for _, _, audio in generator]
    full_audio = np.concatenate(chunks)
    elapsed = time.time() - start
    return full_audio, elapsed

@app.get("/")
def root():
    return {"message": "TTS Quantization Demo API", "endpoints": ["/synthesize", "/compare", "/results"]}

@app.get("/synthesize")
def synthesize(text: str = "Hello from Susani's TTS demo.", model: str = "baseline"):
    pipeline = quantized_pipeline if model == "quantized" else baseline_pipeline
    audio, elapsed = run_tts(pipeline, text)

    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format='WAV')
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"X-Inference-Time": f"{elapsed:.3f}s", "X-Model": model}
    )

@app.get("/compare")
def compare(text: str = "Hello from Susani's TTS demo."):
    _, baseline_time = run_tts(baseline_pipeline, text)
    _, quantized_time = run_tts(quantized_pipeline, text)

    return JSONResponse({
        "text": text,
        "baseline": {"latency_s": round(baseline_time, 3)},
        "quantized": {"latency_s": round(quantized_time, 3)},
        "difference_ms": round((quantized_time - baseline_time) * 1000, 1),
        "note": "Quantized model is smaller (262MB vs 312MB) with tradeoffs on Apple Silicon MPS"
    })

@app.get("/results")
def results():
    return {
        "model": "Kokoro TTS",
        "hardware": "Apple Silicon (MPS)",
        "baseline_size_mb": 312.08,
        "quantized_size_mb": 262.38,
        "size_reduction_pct": 16.0,
        "baseline_latency_s": 0.75,
        "quantized_latency_s": 0.87,
        "finding": "Dynamic PTQ reduces size by 16% but adds latency on ARM — consistent with known PyTorch behavior on Apple Silicon"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
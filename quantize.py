import torch
import soundfile as sf
import time
import os
import numpy as np
from kokoro import KPipeline

torch.backends.quantized.engine='qnnpack'

os.makedirs("outputs",exist_ok=True)

with open("outputs/baseline_time.txt") as f:
    baseline_avg=float(f.read())

ppl = KPipeline(lang_code='a')

def get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024*1024)

original_size = get_model_size_mb(ppl.model)
print(f"original model size:{original_size:.2f} MB")

ppl.model = torch.quantization.quantize_dynamic(
    ppl.model,
    {torch.nn.Linear},
    dtype=torch.qint8,
    inplace=True
)

quantized_size = get_model_size_mb(ppl.model)
print(f"quantized model size:{quantized_size:.2f} MB")

texts = [
    "Hello, my name is Susani and I am a machine learning engineer.",
    "Text to speech synthesis converts written text into spoken audio.",
    "Quantization reduces model size while maintaining high performance."
]

times = []
for i,text in enumerate(texts):
    start = time.time()

    gen = ppl(text,voice="af_nova",speed=1.0)
    chunks=[]

    for gs,ps,audio in gen:
        chunks.append(audio)

    full_audio = np.concatenate(chunks)
    elapsed=time.time()-start
    times.append(elapsed)

    sf.write(f"outputs/quantized_{i+1}.wav",full_audio,24000)
    print(f"sample {i+1}: {elapsed:.2f}s -> outputs/quantized_{i+1}.wav")

quant_avg = sum(times)/len(times)

with open("outputs/results.txt","w") as f:
    f.write(f"original_size={original_size:.2f}\n")
    f.write(f"quantized_size={quantized_size:.2f}\n")
    f.write(f"baseline_avg={baseline_avg:.2f}\n")
    f.write(f"quantized_avg={quant_avg:.2f}\n")





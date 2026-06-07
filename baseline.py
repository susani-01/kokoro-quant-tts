import torch
import soundfile as sf
import time
import os
from kokoro import KPipeline

os.makedirs("outputs",exist_ok=True)
ppl = KPipeline(lang_code='a')

texts = [
    "Hello,my name is Susani and i'm a machine language engineer.",
    "text to speech convertion",
    "Quantization reduces model size while maintaining high performance."
]

times = []

for i, text in enumerate(texts):
    start = time.time()
    gener = ppl(text,voice='af_heart',speed=1.0)

    chunks=[]
    for gs,ps,audio in gener:
        chunks.append(audio)

    import numpy as np
    full_audio = np.concatenate(chunks)

    elapsed = time.time() - start
    times.append(elapsed)

    sf.write(f"outputs/baseline_{i+1}.wav",full_audio,24000)
    print(f"sample {i+1}: {elapsed:.2f}s -> outputs/baseline_{i+1}.wav")

avg = sum(times) / len(times)
print(f"Average inference time: {avg:.2f}s")

with open("outputs/baseline_time.txt","w") as f:
    f.write(str(avg))








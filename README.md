# Kokoro Quant TTS

A lightweight Text-to-Speech (TTS) project that demonstrates **model quantization** for the **Kokoro TTS** model, enabling faster inference and a reduced memory footprint while preserving speech quality.

## Overview

This repository provides a simple workflow for:
- Running the original (baseline) Kokoro TTS model.
- Applying quantization to optimize the model.
- Serving the optimized model for text-to-speech inference.

---

## Features

-  Baseline Kokoro TTS inference
-  Model quantization for optimized deployment
-  Lightweight local inference server
-  Simple and easy-to-understand codebase

---

## Repository Structure

```text
.
├── baseline.py      # Run inference using the original model
├── quantize.py      # Quantize the model
├── serve.py         # Launch the local TTS server
├── requirements.txt # Python dependencies
└── README.md
```


## Project Goal

The objective of this project is to explore the impact of model quantization on neural text-to-speech systems by comparing the original and optimized versions in terms of:

- Model size
- Inference speed
- Deployment efficiency
- Speech quality

---

## Results

The quantized model was evaluated against the original Kokoro TTS model to measure the impact on model size and inference performance.

| Metric | Baseline | Quantized |
|----------|----------|------------|
| Model Size (MB) | 312.08 | 262.38 |
| Average Inference Time (s) | 0.75 | 0.87 |

### Observations

- The quantized model reduces the storage footprint by approximately **15.9%** (312.08 MB → 262.38 MB).
- Quantization introduces a small increase in average inference time in this implementation (**0.75 s → 0.87 s**).
- The primary benefit of this optimization is the reduced model size, which can simplify deployment and lower memory requirements.

> **Note:** Performance may vary depending on the hardware platform, ONNX Runtime version, and quantization strategy used.



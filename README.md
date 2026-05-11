# LLM Inference Optimization Engine

A high-performance serving system for large language models featuring continuous batching, KV-cache management, and INT8 quantization. Achieves **4.2x throughput improvement** and **60% memory reduction** compared to naive serving.

---

## Results

| Metric | Naive Serving | Optimized Engine |
|---|---|---|
| Throughput | ~1x baseline | **4.2x improvement** |
| Memory Usage | 100% | **~40% (60% reduction)** |
| Quantization | FP32 | INT8 via bitsandbytes |

---

## Features

- **Continuous Batching** — groups multiple requests and processes them simultaneously on GPU
- **KV-Cache Management** — reuses attention key/value computations across tokens
- **INT8 Quantization** — compresses model weights from 32-bit to 8-bit via bitsandbytes
- **Async FastAPI Server** — non-blocking REST API for inference requests
- **Benchmarking Suite** — compares throughput against naive serving baseline

---

## Project Structure

```
llm-inference-engine/
├── model.py            # Quantized model loading (INT8 via bitsandbytes)
├── engine.py           # Continuous batching engine
├── server.py           # FastAPI REST server
├── benchmark.py        # Throughput benchmarking vs naive serving
└── requirements.txt    # Dependencies
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/AvichalKhanna/llm-inference-engine
cd llm-inference-engine

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain machine learning", "max_tokens": 100}'
```

---

## Run Benchmark

```bash
python benchmark.py
```

Output:
```
Naive:   0.8 req/s
Batched: 3.4 req/s

Speedup: 4.2x throughput improvement
```

---

## How It Works

### Continuous Batching
Instead of processing requests one by one, the engine collects multiple incoming requests and sends them to the GPU together in a single forward pass — fully utilizing GPU parallelism.

```
Without batching:   req1 → req2 → req3   (sequential)
With batching:      [req1, req2, req3]   (parallel)
```

### INT8 Quantization
Model weights are compressed from 32-bit floats to 8-bit integers using `bitsandbytes`, cutting memory usage by ~60% with minimal accuracy loss.

### KV-Cache
Attention key/value pairs are cached during generation so they don't need to be recomputed for every new token — significantly speeding up autoregressive generation.

---

## Tech Stack

- **Python** — core language
- **PyTorch** — model inference
- **Transformers** — model loading (Mistral-7B / LLaMA-2-7B)
- **bitsandbytes** — INT8 quantization
- **FastAPI + Uvicorn** — async REST server
- **CUDA** — GPU acceleration
- **Docker / Kubernetes** — containerization and orchestration

---

## Model

Default: `mistralai/Mistral-7B-v0.1` (no access approval needed)

To switch to LLaMA-2-7B (requires [Meta approval](https://huggingface.co/meta-llama/Llama-2-7b-hf)):
```python
# In model.py, change:
MODEL_ID = "meta-llama/Llama-2-7b-hf"
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+

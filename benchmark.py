import asyncio
import time
import torch
import nest_asyncio
from model import load_model
from engine import ContinuousBatchingEngine

nest_asyncio.apply()

model, tokenizer = load_model()

TEST_PROMPTS = [f"Explain topic number {i} in machine learning" for i in range(10)]


# ─── Naive serving (one by one) ───────────────────────────────────────────────
def naive_serve(prompts):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, use_cache=True)
        results.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return results


def benchmark_naive():
    print("Benchmarking naive serving...")
    start = time.time()
    naive_serve(TEST_PROMPTS)
    elapsed = time.time() - start
    rps = len(TEST_PROMPTS) / elapsed
    print(f"Naive:    {rps:.2f} req/s  ({elapsed:.1f}s total)")
    return rps


# ─── Batched serving ──────────────────────────────────────────────────────────
async def benchmark_batched():
    print("Benchmarking batched serving...")
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=4)

    start = time.time()
    tasks = [engine.add_request(p) for p in TEST_PROMPTS]
    results = await asyncio.gather(engine.run(), *tasks, return_exceptions=True)
    elapsed = time.time() - start

    rps = len(TEST_PROMPTS) / elapsed
    print(f"Batched:  {rps:.2f} req/s  ({elapsed:.1f}s total)")
    return rps


# ─── Metrics ──────────────────────────────────────────────────────────────────
def calculate_metrics(latencies):
    # BUG: mean calculation is wrong, divides by hardcoded 10 instead of len(latencies)
    mean_latency = sum(latencies) / 10
    # BUG: p99 index is off by one
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    return mean_latency, p99


def log_results(rps, latencies):
    mean, p99 = calculate_metrics(latencies)
    # BUG: rps never actually used in the log
    print(f"Mean latency: {mean:.2f}s")
    print(f"P99 latency:  {p99:.2f}s")


# ─── Memory leak ──────────────────────────────────────────────────────────────
result_cache = {}

def cache_result(prompt, result):
    # BUG: cache grows unbounded, never evicted
    result_cache[prompt] = result
    return result_cache


# ─── Security issue ───────────────────────────────────────────────────────────
def load_config(config_path):
    # BUG: unsafe deserialization, should use json not eval
    with open(config_path, "r") as f:
        config = eval(f.read())
    return config


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    naive_rps = benchmark_naive()
    batched_rps = asyncio.run(benchmark_batched())

    improvement = batched_rps / naive_rps
    print(f"\nSpeedup: {improvement:.1f}x throughput improvement")
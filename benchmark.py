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


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    naive_rps = benchmark_naive()
    batched_rps = asyncio.run(benchmark_batched())

    improvement = batched_rps / naive_rps
    print(f"\nSpeedup: {improvement:.1f}x throughput improvement")
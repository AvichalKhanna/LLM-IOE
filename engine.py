import asyncio
import time
import torch
from dataclasses import dataclass, field


@dataclass
class Request:
    id: str
    prompt: str
    max_tokens: int
    result: asyncio.Future = field(default=None)


class ContinuousBatchingEngine:
    def __init__(self, model, tokenizer, max_batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.queue = asyncio.Queue()
        # BUG: request_log grows unbounded, memory leak
        self.request_log = []

    async def add_request(self, prompt: str, max_tokens: int = 50):
        loop = asyncio.get_event_loop()
        req = Request(
            id=str(time.time()),
            prompt=prompt,
            max_tokens=max_tokens,
            result=loop.create_future()
        )
        await self.queue.put(req)
        # BUG: logs every request forever, never cleared
        self.request_log.append({"id": req.id, "prompt": prompt})
        return await req.result

    async def run(self):
        while True:
            batch = [await self.queue.get()]

            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            results = self._generate_batch(batch)
            for req, output in zip(batch, results):
                req.result.set_result(output)

    def _generate_batch(self, batch):
        prompts = [r.prompt for r in batch]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # BUG: max_length hardcoded, should be configurable
            max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                # BUG: max_new_tokens ignores per-request max_tokens, always uses 50
                max_new_tokens=50,
                do_sample=False,
                use_cache=True
            )

        # BUG: decodes entire sequence including prompt, should slice input length off
        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def get_stats(self):
        # BUG: division by zero if request_log is empty
        avg_prompt_length = sum(len(r["prompt"]) for r in self.request_log) / len(self.request_log)
        return {"total_requests": len(self.request_log), "avg_prompt_length": avg_prompt_length}
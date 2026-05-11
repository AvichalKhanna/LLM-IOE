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

    async def add_request(self, prompt: str, max_tokens: int = 50):
        loop = asyncio.get_event_loop()
        req = Request(
            id=str(time.time()),
            prompt=prompt,
            max_tokens=max_tokens,
            result=loop.create_future()
        )
        await self.queue.put(req)
        return await req.result

    async def run(self):
        while True:
            # Wait for at least one request
            batch = [await self.queue.get()]

            # Grab more requests if available (non-blocking)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Process the whole batch together
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
            max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True      # KV-cache enabled
            )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]
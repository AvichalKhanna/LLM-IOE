import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio

from model import load_model
from engine import ContinuousBatchingEngine

nest_asyncio.apply()

app = FastAPI(title="LLM Inference Engine")

model, tokenizer = load_model()
engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=4)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.on_event("startup")
async def startup():
    asyncio.create_task(engine.run())
    print("Batching engine started!")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    result = await engine.add_request(request.prompt, request.max_tokens)
    return {"output": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
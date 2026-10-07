"""FastAPI service: /ask (troubleshooting assistant), /block/{id} (detector verdict), /health."""
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag

app = FastAPI(title="Network Incident Troubleshooting Assistant")
CHAIN, BACKEND = rag.build_chain()


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


@app.post("/ask")
def ask(body: Ask):
    t0 = time.perf_counter()
    out = rag.ask(body.question, CHAIN)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get("/block/{block_id}")
def block(block_id: str):
    ctx = rag.structured_context(block_id)
    b = ctx["blocks"][0] if ctx["blocks"] else None
    if not b or not b.get("found"):
        raise HTTPException(404, f"{block_id} not in indexed logs")
    return {"block_id": block_id, "anomalous": rag.block_verdict(b), **b}

"""
Exercise 2 API contract -- your system MUST expose exactly this interface.

The blind evaluation calls POST /ask on your endpoint with an AskRequest and
expects an AskResponse. Fields you don't fill (e.g. cost_usd) simply score
worse on the efficiency component; fields with wrong types fail validation.

Run this stub as-is to see the contract in action:

    uvicorn contract:app --port 8000
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "האם הביטוח מכסה נזק מפגיעת ברק?"}'

Replace `answer_question` with your actual system. Do not change the models.
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import rag_runner


class AskRequest(BaseModel):
    question: str = Field(..., description="Customer question, usually Hebrew")
    session_id: Optional[str] = Field(None, description="For multi-turn context (optional)")


class Citation(BaseModel):
    file: str = Field(..., description="Source document path or URL")
    page: Optional[int] = Field(None, description="1-based page number for PDFs")
    quote: Optional[str] = Field(None, description="The supporting passage (optional but persuasive)")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The answer, in the language of the question")
    citations: List[Citation] = Field(default_factory=list)
    domain: Optional[str] = Field(None, description="Routed insurance domain, e.g. 'travel'")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = Field(None, description="Estimated $ cost of answering this question")


# Built once at startup and reused across requests (the vector db + embedder are
# expensive to construct). Populated by the lifespan handler below.
rag_system: Optional["rag_runner.RagSystem"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    rag_system = rag_runner.build_rag_system()  # parse -> chunk -> embed -> index (disk-cached)
    print("RAG system ready; serving /ask")
    yield


app = FastAPI(title="APEX Exercise 2 -- Harel Support Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if rag_system is not None else "starting"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = rag_system.answer(req.question)
    return AskResponse(
        answer=result["answer"],
        citations=[Citation(file=c["file"], page=c["page"]) for c in result["citations"]],
        domain=result["domain"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
    )

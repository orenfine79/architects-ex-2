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
import time
from typing import List, Optional

import litellm
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_runner import answer_questions, init_service


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


app = FastAPI(title="APEX Exercise 2 -- Harel Support Agent")

init_service()  # build the RAG retrieval stack at startup, not on the first /ask


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.time()
    svc = init_service()
    result = next(answer_questions(svc.collection, svc.embedder, [req.question],
                                   embed_model=svc.embed_model))
    hits = result.hits
    citations = [Citation(file=c.file, page=c.page) for c in result.citations]

    try:
        cost = litellm.completion_cost(completion_response=result.response)
    except Exception:  # custom endpoints (Token Factory, vLLM) have no price table
        cost = None

    return AskResponse(
        answer=result.answer,
        citations=citations,
        domain=hits[0]["domain"] if hits else None,
        # top-hit retrieval similarity, zeroed when the model cited nothing
        confidence=min(max(float(hits[0]["score"]), 0.0), 1.0) if citations else 0.0,
        latency_ms=(time.time() - t0) * 1000,
        cost_usd=cost,
    )

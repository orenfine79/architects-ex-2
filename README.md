# Exercise 2 — Harel Insurance Support Agent

Read **`exercise2_customer_support_agent.md`** for the full exercise.

## What's here

| Path | What |
|---|---|
| `exercise2_customer_support_agent.md` | The exercise |
| `reference_questions.json` | Dev Q&A set: questions labeled easy/medium/hard, ground-truth answers and file+page citations |
| `contract.py` | The FastAPI `/ask` schema your system must expose (grading calls it) |
| `baseline_runner.py` | Stage 1: run the questions through a bare model, answers JSONL out |
| `submit_runner.py` | Batch-asks your `/ask` endpoint → answers JSONL (used for final submission) |
| `tf_client.py` | Minimal Token Factory client with per-call cost estimate (shared key — play fair) |

There is deliberately no evaluation script here: **building your own harness is
a Stage 1 deliverable** — the exercise page documents exactly what it must
measure (relevance, hallucination rate, citation accuracy, latency).

## Quickstart

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=...               # the shared course Token Factory key
export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
export OPENAI_API_KEY=$NEBIUS_API_KEY

# Stage 1: baseline answers, then score them with YOUR harness
python baseline_runner.py --model google/gemma-3-27b-it
```

The document corpus: `python get_corpus.py` downloads the frozen snapshot
from the public HF dataset `orik/apex-ex2-harel-corpus` into a local `corpus/`
dir (gitignored). Ground-truth answers are anchored to that snapshot, not the
live site.

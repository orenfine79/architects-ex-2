"""
Stage 1 baseline: run the dev questions straight through a bare model with
NO retrieval, producing an answers file the eval harness can score.

    export OPENAI_API_KEY=...  # paste in nebius API key
    export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1  # Use token factory
    python baseline_runner.py --model deepseek-ai/DeepSeek-V4-Pro
    # then score baseline_answers.jsonl with YOUR evaluation harness (Stage 1)

Calls go through litellm: a bare model name goes to OpenAI; set
OPENAI_BASE_URL for any OpenAI-compatible endpoint (Token Factory, a local
vLLM server, ...); provider-prefixed models ("anthropic/...", "gemini/...")
work with the matching key env var. Try --system-prompt variants and watch
how the failure profile (not just the score) changes.
"""
import argparse
import json
import os
import time

import litellm
from dotenv import load_dotenv

from eval_harness import run_evaluation

# Loads the variables from .env into the environment
load_dotenv() 

MODEL = "deepseek-ai/DeepSeek-V4-Pro"

# V1 
V1_SYSTEM = """
You are a customer-support assistant for Harel Insurance (Israel).
Answer the customer's question in the language it was asked.
If you cite a source, cite the exact document and page.
"""

# V2
V2_SYSTEM = """
You are a customer-support assistant for Harel Insurance (Israel).
Answer the customer's question in the language it was asked.
If you cite a source, cite the exact document and page.
Only answer when you're completely confident. If you're not completely sure, say that you don't know.
"""

V3_SYSTEM = """
You are a customer-support assistant for the Israeli insurance company Harel.
Answer the customer's question in the language it was asked. Your answers should be brief and concise.  
If you're presented with a yes/no question, start by explicitly stating "Yes" or "No". 
If you cite a source, cite the exact document and page.
Only answer when you're completely confident. If you're not completely sure, only write: "I don't know".
"""

V4_SYSTEM = """
You are a highly professional legal assistant for Harel Insurance (Israel), specializing in Harel insurance policy terms, service agreements
and claims procedures.
Your goal is to provide accurate, clear, and structured answers to the various inquiries.
Answer in the exact language the user used to ask the question
If you cite a source, cite the exact document and page.
If you are not 100% certain of the answer based on standard Harel policies state clearly that you do not have the exact information to answer this question.
Focus on answering the question only. Avoid asking questions back, avoid nicities (e.g. "Thanks for asking"), avoid tasking the user ("check your policy for...", 
"speak to...", etc.), avoid proposing additional tasks ("would you like me to...").
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--system-prompt", default=V4_SYSTEM)
    ap.add_argument("--out", default="baseline_answers.jsonl")
    ap.add_argument("--eval", action="store_true",
                    help="after generating answers, score them with the eval harness")
    ap.add_argument("--eval-out", default="evaluation.jsonl",
                    help="where to write per-item eval results (with --eval)")
    args = ap.parse_args()

    # routing: OPENAI_BASE_URL forces the openai/ route to that endpoint,
    # whatever the model id looks like (TF ids contain "/")
    model, kwargs = args.model, {}
    base = os.getenv("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"

    questions = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(questions, dict):  # staff sets wrap the list in {"questions": [...]}
        questions = questions["questions"]
    with open(args.out, "w", encoding="utf-8") as out:
        for q in questions:
            t0 = time.time()
            resp = litellm.completion(model=model, messages=[
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": q["question"]}],
                timeout=120, **kwargs)
            rec = {"id": q["id"],
                   "answer": resp.choices[0].message.content,
                   "citations": [],  # the model has no documents -- that's the point
                   "latency_ms": (time.time() - t0) * 1000,
                   "tokens": {"prompt": resp.usage.prompt_tokens,
                              "completion": resp.usage.completion_tokens}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {rec['answer'][:70]!r}... ({rec['latency_ms']:.0f} ms)")
    
    
    print(f"\nwrote {args.out} -- now score it with your evaluation harness")

    if args.eval:
        print("\nscoring answers with the eval harness ...")
        run_evaluation(questions, args.out, args.eval_out)


if __name__ == "__main__":
    main()

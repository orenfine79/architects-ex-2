"""
Nebius Token Factory client. Everyone shares ONE course API key with a shared
balance -- play fair: watch the cost estimate this prints and don't burn the
pool. Set NEBIUS_API_KEY, then:

    from tf_client import chat
    reply = chat([{"role": "user", "content": "..."}], model="deepseek-ai/DeepSeek-V4-Pro")

    python tf_client.py --model Qwen/Qwen3-32B "מה מכסה ביטוח דירה?"
"""
import argparse
import os
import sys

import litellm
from dotenv import load_dotenv

# Loads the variables from .env into the environment
load_dotenv() 

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
# the API doesn't return cost, so we estimate: $/1M tokens (in, out),
# priced like a stronger model (DeepSeek-class). Real cost may be lower.
EST_PRICE = (0.5, 2.0)

def chat(messages, model, max_tokens=1024, temperature=0.2, quiet=False, **kw):
    """OpenAI-compatible chat completion via litellm; prints a per-call cost estimate."""
    # key = os.environ.get("NEBIUS_API_KEY") or sys.exit("NEBIUS_API_KEY not set")
    key = os.getenv("NEBIUS_API_KEY") or sys.exit("NEBIUS_API_KEY not set")
    # openai/ prefix: TF model ids contain "/" which litellm would misread as
    # a provider; force the openai-compatible route to BASE_URL instead
    resp = litellm.completion(model=f"openai/{model}", api_base=BASE_URL, api_key=key,
                              messages=messages, max_tokens=max_tokens,
                              temperature=temperature, timeout=180, **kw)
    u = resp.usage
    cost = (u.prompt_tokens * EST_PRICE[0] + u.completion_tokens * EST_PRICE[1]) / 1e6
    if not quiet:
        print(f"[tf_client] {u.prompt_tokens}+{u.completion_tokens} tokens "
              f"~${cost:.4f}", file=sys.stderr)
    return resp.choices[0].message.content


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    args = ap.parse_args()
    print(chat([{"role": "user", "content": args.prompt}], model=args.model))

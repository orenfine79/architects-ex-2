import argparse, json, os, re, time
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from tf_client import chat
import pandas as pd

# Loads the variables from .env into the environment
load_dotenv() 

# EVAL_SYSTEM_PROMPT = """
# You are an expert evaluator assessing the quality, accuracy, and truthfulness of an AI-generated answer compared to a verified ground truth answer.

# Your task is to analyze the Candidate Answer against the Ground Truth Answer for the given Question and categorize its behavior precisely.

# [Classification Rules]
# 1. REFUSAL: Check if the candidate answer explicitly states it does not know, cannot answer, or lacks the necessary context to answer (e.g., "I don't know," "I am sorry, I cannot answer based on the text").
#    - If the candidate refuses, set "is_refusal" to true.
#    - A refusal is NEVER a hallucination.

# 2. HALLUCINATION: Check if the candidate answer makes a confident factual claim that directly contradicts the Ground Truth Answer, or introduces speculative, fabricated "facts" entirely unsupported by the ground truth.
#    - If it makes a false or contradictory claim confidently, set "is_hallucination" to true.
#    - Note: A minor omission or a difference in phrasing is NOT a hallucination. It must be a factual contradiction or fabrication.

# [Similarity Score Guidelines]
# 5 - Perfect: Factual match, handles nuances perfectly.
# 4 - High: Covers main facts, minor omissions, zero contradictions.
# 3 - Partial: Gets some facts right, misses key context. 
# 2 - Low / Safe Refusal: The candidate safely stated it didn't know (Refusal), OR missed almost all core facts.
# 1 - Fail / Hallucination: The candidate confidently stated incorrect or contradictory information.

# [Output Format]
# Provide your evaluation in valid JSON format exactly as follows:
# {
#   "reasoning": "A concise explanation detailing why you chose the score and whether a refusal or hallucination occurred.",
#   "similarity_score": <int between 1 and 5>,
#   "is_refusal": <boolean true or false>,
#   "is_hallucination": <boolean true or false>
# }
# """

EVAL_SYSTEM_PROMPT = """
Evaluate the Candidate Answer against the Ground Truth.

Return ONLY a single valid JSON object (no markdown fences, no comments, no text
before or after) with exactly these keys:
- "reasoning": string. Brief justification. Keep it on one line and do not use
  double quotes inside it.
- "similarity_score": integer 1-5. 5: Perfect, 4: Minor omission, 3: Partial,
  2: Refusal/Low, 1: Hallucination.
- "is_refusal": boolean. true if the candidate explicitly states it cannot
  answer / doesn't know.
- "is_hallucination": boolean. true if the candidate confidently contradicts the
  Ground Truth (minor omissions do not count).

Example:
{"reasoning": "Covers all key facts with no contradictions.", "similarity_score": 5, "is_refusal": false, "is_hallucination": false}
"""

MODEL = "deepseek-ai/DeepSeek-V4-Pro"

PARSED_DIR = Path("parsed_corpus")

CITATION_JUDGE_SYSTEM_PROMPT = """
You are verifying whether a cited source passage actually backs up a generated
answer -- i.e. that the citation is not fabricated or irrelevant. You are given
the Question, the Generated Answer (produced by an AI, which cited this source),
and the Cited Passage (the exact text of the page the answer cited).

Decide whether the Cited Passage contains the information needed to support the
claims made in the Generated Answer. It does NOT need to state the answer
verbatim -- it is enough that the facts in the passage back up what the answer
claims. Answer false if the passage is irrelevant or does not contain the
information the generated answer relies on.

Return ONLY a single valid JSON object (no markdown fences, no comments, no text
before or after) with exactly these keys:
- "reasoning": string. Brief justification on one line, no double quotes inside.
- "supported": boolean. true if the passage supports the Generated Answer,
  false if it is irrelevant or does not contain the needed information.

Example:
{"reasoning": "The passage states filing with an institution does not stop the limitation period, which is exactly what the generated answer claims.", "supported": true}
"""


def run_evaluation(questions, answers, out="evaluation.jsonl",
                   model=MODEL, system_prompt=EVAL_SYSTEM_PROMPT):
    """Score an answers file against reference questions and write per-item results.

    Callable from other modules (e.g. baseline_runner.py) or the CLI.

    Args:
        questions: path to the reference questions json, or an already-loaded list.
        answers:   path to a jsonl answers file, or an already-loaded list of records.
        out:       path to write per-item evaluation results (jsonl).
        model:     eval model id passed to tf_client.chat.
        system_prompt: eval system prompt.

    Returns:
        A summary dict with averages and the hallucination count/rate.
    """
    if isinstance(questions, str):
        questions = json.load(open(questions, encoding="utf-8"))
    if isinstance(questions, dict):  # staff sets wrap the list in {"questions": [...]}
        questions = questions["questions"]

    if isinstance(answers, str):
        answers = pd.read_json(answers, lines=True).to_dict(orient="records")
    answers_by_id = {a["id"]: a for a in answers}

    hallucination_count = 0; total_latency = 0; total_questions = 0; total_answer_score = 0

    with open(out, "w", encoding="utf-8") as out_f:
        for q in questions:
            total_questions += 1
            result = eval_harness(
                id = q["id"],
                question = q["question"],
                generated_answer = answers_by_id[q["id"]]["answer"],
                ground_truth_answer = q["ground_truth_answer"],
                generated_citations = answers_by_id[q["id"]]["citations"],
                ground_truth_citations = q["ground_truth_sources"],
                latency = answers_by_id[q["id"]]["latency_ms"],
                model = model,
                system_prompt = system_prompt,
            )
            print(
                f"id: {result['id']}: answer_score: {result['answer_score']}, "
                f"citation_score: {result['citation_score']}, "
                f"Hallucinate: {True if result['answer_score'] == 1 else False}"
            )

            total_answer_score += result["answer_score"]
            if result["hallucinate"]:
                hallucination_count += 1
            total_latency += result["latency"]

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        print("avg_score: ", total_answer_score / total_questions)
        print("avg latency: ", total_latency / total_questions)
        print(f"hallucination_rate: {hallucination_count / total_questions} ({hallucination_count} / {total_questions})")
        print("=" * 40)

    summary = {
        "count": total_questions,
        "avg_answer_score": total_answer_score / total_questions if total_questions else 0,
        "hallucinations": hallucination_count,
        "hallucination_rate": hallucination_count / total_questions if total_questions else 0,
        "avg_latency_ms": total_latency / total_questions if total_questions else 0,
    }
    print(f"\nwrote {out} -- avg_score={summary['avg_answer_score']:.2f} "
          f"hallucinations={hallucination_count}/{total_questions} "
          f"avg_latency={summary['avg_latency_ms']:.0f} ms")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--system-prompt", default=EVAL_SYSTEM_PROMPT)
    ap.add_argument("--answers", default="baseline_answers_v4.jsonl")
    ap.add_argument("--out", default="evaluation.jsonl")
    args = ap.parse_args()

    run_evaluation(args.questions, args.answers, args.out,
                   model=args.model, system_prompt=args.system_prompt)


def parse_eval_json(reply: str) -> dict:
    """Best-effort parse of the eval model's JSON reply.

    LLM judges routinely wrap output in ```json fences, add `//` comments, or
    leave stray text around the object. We strip those, isolate the outermost
    {...}, and try json.loads. If that still fails (e.g. an unescaped quote in
    the free-text 'reasoning'), we regex out the fields we actually need so one
    messy reply doesn't abort the whole run.
    """
    text = reply.strip()

    # strip markdown code fences: ```json ... ``` or ``` ... ```
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # isolate the outermost JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # drop // line comments and /* block */ comments (invalid JSON)
    no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    no_comments = re.sub(r"//[^\n\r]*", "", no_comments)

    try:
        return json.loads(no_comments)
    except json.JSONDecodeError:
        pass

    # last resort: pull the fields out individually
    score = re.search(r'"similarity_score"\s*:\s*(\d+)', text)
    if not score:
        raise ValueError(f"could not parse eval reply: {reply!r}")

    def _bool(field):
        m = re.search(rf'"{field}"\s*:\s*(true|false)', text)
        return m is not None and m.group(1) == "true"

    reason = re.search(r'"reasoning"\s*:\s*"(.*?)"', text, re.DOTALL)
    return {
        "reasoning": reason.group(1) if reason else "",
        "similarity_score": int(score.group(1)),
        "is_refusal": _bool("is_refusal"),
        "is_hallucination": _bool("is_hallucination"),
    }


def eval_harness(id, question, generated_answer, ground_truth_answer, generated_citations, ground_truth_citations, latency,
                 model=MODEL, system_prompt=EVAL_SYSTEM_PROMPT):
            # user_message = f"""
            # [Inputs]
            # Question: {question}
            # Ground Truth Answer: {ground_truth_answer}
            # Candidate Answer: {generated_answer}
            # """

            user_message = f"""
            [Inputs]
            Ground Truth Answer: {ground_truth_answer}
            Candidate Answer: {generated_answer}
            """

            reply = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=model,
                temperature=0,
                quiet=True
            )

            answer_eval = parse_eval_json(reply)
            answer_score = answer_eval["similarity_score"]
            
            citation_eval = evaluate_citations(
                question=question,
                generated_answer=generated_answer,
                candidate_citations=generated_citations,
                model=model,
            )
            citation_score = citation_eval["score"]
            
            return {
                "id": id, 
                "answer_score": answer_score, 
                "citation_score": citation_score, 
                "hallucinate": True if answer_score==1 else False, 
                "latency": latency,
            }
            
            # rec = {"id": q["id"],
            #        "score": reply,
            #     #    "refusal": ,
            #     #    "hallucinate": ,
            #     #    "reason":
            #        "citations": [],  # the model has no documents -- that's the point
            #        "latency_ms": (time.time() - t0) * 1000,
            #        "tokens": {"prompt": resp.usage.prompt_tokens,
            #                   "completion": resp.usage.completion_tokens}}
            # out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            # print(f"{q['id']}: {rec['answer'][:70]!r}... ({rec['latency_ms']:.0f} ms)")
    # print(f"\nwrote {args.out} -- now score it with your evaluation harness")


def normalize_path(path) -> str:
    """Normalizes file paths to handle trailing slashes, whitespace, and OS-specific slashes."""
    if not isinstance(path, str):
        return ""
    # Replace backslashes with forward slashes, strip whitespace and leading/trailing slashes
    return path.replace("\\", "/").strip().strip("/")


def normalize_page(page):
    """Normalizes page numbers to integers to handle '32', 32, or 32.0 gracefully."""
    if page is None:
        return None
    try:
        return int(float(page))
    except (ValueError, TypeError):
        return str(page).strip()


@lru_cache(maxsize=None)
def _load_parsed_doc(file: str):
    """Loads a parsed corpus document (list of {page, text}) for a citation file.

    Cached because the same document is cited many times across a run. Returns
    the 'pages' list, or None if the parsed file is missing/unreadable.
    """
    rel = normalize_path(file)
    path = PARSED_DIR / f"{rel}.json"
    if not path.exists():
        return None
    try:
        return json.load(open(path, encoding="utf-8")).get("pages", [])
    except (json.JSONDecodeError, OSError):
        return None


def load_citation_text(file, page) -> str:
    """Returns the text of the cited page from the parsed corpus.

    `page` is matched leniently ('32', 32, 32.0 all match). When page is None
    (e.g. a web source parsed as a single page), all pages are concatenated.
    Returns "" if the document or page can't be found.
    """
    pages = _load_parsed_doc(file)
    if not pages:
        return ""

    want = normalize_page(page)
    if want is None:
        return "\n\n".join(p.get("text", "") for p in pages).strip()

    for p in pages:
        if normalize_page(p.get("page")) == want:
            return (p.get("text") or "").strip()
    return ""


def parse_citation_json(reply: str) -> dict:
    """Parses the citation judge's JSON reply into {reasoning, supported}."""
    text = reply.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        obj = json.loads(text)
        return {"reasoning": str(obj.get("reasoning", "")),
                "supported": bool(obj.get("supported", False))}
    except json.JSONDecodeError:
        pass

    m = re.search(r'"supported"\s*:\s*(true|false)', text)
    reason = re.search(r'"reasoning"\s*:\s*"(.*?)"', text, re.DOTALL)
    return {
        "reasoning": reason.group(1) if reason else "",
        "supported": m is not None and m.group(1) == "true",
    }


def judge_citation(question, generated_answer, citation_text,
                   model=MODEL, system_prompt=CITATION_JUDGE_SYSTEM_PROMPT) -> dict:
    """Asks the LLM judge whether one cited passage supports the generated answer."""
    user_message = f"""
    [Inputs]
    Question: {question}
    Generated Answer: {generated_answer}
    Cited Passage: {citation_text}
    """
    reply = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        temperature=0,
        quiet=True,
    )
    return parse_citation_json(reply)


def evaluate_citations(question, generated_answer, candidate_citations,
                       model=MODEL, system_prompt=CITATION_JUDGE_SYSTEM_PROMPT) -> dict:
    """LLM-as-judge citation check.

    For each generated citation, fetches the cited page text from the parsed
    corpus and asks the judge whether that passage actually supports the
    generated answer (i.e. the citation is grounded, not fabricated). Scoring
    is binary: 1.0 if at least one citation is supporting, else 0.0 (and 0.0
    when there are no citations at all).
    """
    details = []
    for cite in candidate_citations:
        file = cite.get("file")
        page = cite.get("page")
        text = load_citation_text(file, page)

        if not text:  # can't locate the cited page -> it can't support anything
            details.append({"file": file, "page": page, "supported": False,
                            "reasoning": "cited page not found in parsed corpus",
                            "text_found": False})
            continue

        verdict = judge_citation(question, generated_answer, text,
                                 model=model, system_prompt=system_prompt)
        details.append({"file": file, "page": page,
                        "supported": verdict["supported"],
                        "reasoning": verdict["reasoning"],
                        "text_found": True})

    supported_count = sum(1 for d in details if d["supported"])
    score = 1.0 if supported_count > 0 else 0.0

    return {
        "is_fully_correct": score == 1.0,
        "score": score,  # binary: any citation supports the answer?
        "supported_citations_count": f"{supported_count}/{len(candidate_citations)}",
        "details": details,
    }


# ==========================================
# Execution Demo with Your Data
# ==========================================
# if __name__ == "__main__":
#     # Your ground truth structure
#     ground_truth_sources = [
#         {
#             "any_of": [
#                 {
#                     "file": "apartment/files/חוברת-הכללים-לגבי-פוליסות-שנרכשו-לאחר-ה-030917.pdf",
#                     "page": 32,
#                 }
#             ]
#         }
#     ]

#     # Test Case 1: Your exact example answer (Empty Citations) -> Should Fail
#     empty_citations = []
#     print("--- Test Case 1: Empty Citations ---")
#     result_1 = evaluate_citations(empty_citations, ground_truth_sources)
#     print(f"Fully Correct: {result_1['is_fully_correct']}")
#     print(f"Score: {result_1['score']}")

#     print("\n" + "=" * 40 + "\n")

#     # Test Case 2: System generated correct citation (with slight string/int variations) -> Should Pass
#     correct_citations = [
#         {
#             "file": " /apartment/files/חוברת-הכללים-לגבי-פוליסות-שנרכשו-לאחר-ה-030917.pdf ",  # leading/trailing spaces
#             "page": "32",  # represented as string
#         }
#     ]
#     print("--- Test Case 2: Normalized Correct Citations ---")
#     result_2 = evaluate_citations(correct_citations, ground_truth_sources)
#     print(f"Fully Correct: {result_2['is_fully_correct']}")
#     print(f"Score: {result_2['score']}")
#     print(f"Matched details: {result_2['details'][0]['matched_with']}")

if __name__ == "__main__":
    main()

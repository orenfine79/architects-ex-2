import argparse, json, os, re, time
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
    ap.add_argument("--answers", default="baseline_answers_v2.jsonl")
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
            
            citation_eval = evaluate_citations(generated_citations, ground_truth_citations)
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


def check_source_match(candidate, expected) -> bool:
    """Checks if a single candidate citation matches an expected ground truth source."""
    cand_file = normalize_path(candidate.get("file"))
    exp_file = normalize_path(expected.get("file"))

    cand_page = normalize_page(candidate.get("page"))
    exp_page = normalize_page(expected.get("page"))

    return cand_file == exp_file and cand_page == exp_page


def evaluate_citations(candidate_citations, ground_truth_sources) -> dict:
    """Evaluates if the candidate's citations satisfy the ground truth requirements.

    Handles:
      - Single-document questions (1 group)
      - Cross-document questions (multiple groups, requiring one match per group)
      - 'any_of' conditional matches within each group
    """

    total_groups = len(ground_truth_sources)
    group_results = []
    
    # Each 'group' represents a distinct fact/document requirement
    for idx, group in enumerate(ground_truth_sources):
        any_of_list = group.get("any_of", [])
        group_satisfied = False
        matched_by = None

        # Check if ANY of the acceptable sources for this group are cited
        for expected_source in any_of_list:
            for candidate_cite in candidate_citations:
                if check_source_match(candidate_cite, expected_source):
                    group_satisfied = True
                    matched_by = candidate_cite
                    break
            if group_satisfied:
                break

        group_results.append(
            {
                "group_index": idx,
                "satisfied": group_satisfied,
                "matched_with": matched_by,
                "options": any_of_list,
            }
        )

    # Calculate overall score
    satisfied_count = sum(1 for g in group_results if g["satisfied"])
    score = satisfied_count / total_groups if total_groups > 0 else 1.0

    return {
        "is_fully_correct": satisfied_count == total_groups,
        "score": score,  # Fraction of required facts cited
        "satisfied_groups_count": f"{satisfied_count}/{total_groups}",
        "details": group_results,
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

"""Join reference questions, generated answers, and eval scores into one report.

    python build_report.py \
        --questions reference_questions.json \
        --answers   baseline_answers_v4.jsonl \
        --eval      evaluation_v4_1.jsonl \
        --out       eval_report.txt
"""
import argparse, json


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--answers", default="baseline_answers_v4.jsonl")
    ap.add_argument("--eval", default="evaluation.jsonl")
    ap.add_argument("--out", default="eval_report.txt")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    answers = {a["id"]: a for a in load_jsonl(args.answers)}
    evals = {e["id"]: e for e in load_jsonl(args.eval)}

    sep = "=" * 80
    with open(args.out, "w", encoding="utf-8") as out:
        for q in questions:
            qid = q["id"]
            ans = answers.get(qid, {})
            ev = evals.get(qid, {})

            out.write(f"{sep}\n")
            out.write(f"ID: {qid}\n")
            out.write(f"{sep}\n\n")

            out.write("QUESTION:\n")
            out.write(f"{q['question']}\n\n")

            out.write("REFERENCE ANSWER:\n")
            out.write(f"{q['ground_truth_answer']}\n\n")

            out.write("GENERATED ANSWER:\n")
            out.write(f"{ans.get('answer', '[missing]')}\n\n")

            out.write("SCORES:\n")
            out.write(f"  answer_score:   {ev.get('answer_score', 'N/A')}\n")
            out.write(f"  citation_score: {ev.get('citation_score', 'N/A')}\n")
            out.write(f"  Hallucinate:    {ev.get('hallucinate', 'N/A')}\n\n\n")

    print(f"wrote {args.out} ({len(questions)} items)")


if __name__ == "__main__":
    main()

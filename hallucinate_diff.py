"""Find rows where the `hallucinate` field differs between two evaluation jsonl files."""
import json
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec["id"]] = rec
    return rows


def main(file_a, file_b):
    a = load(file_a)
    b = load(file_b)

    all_ids = sorted(set(a) | set(b))
    diffs = []
    for _id in all_ids:
        if _id not in a:
            print(f"[only in B] {_id}")
            continue
        if _id not in b:
            print(f"[only in A] {_id}")
            continue
        ha = a[_id].get("hallucinate")
        hb = b[_id].get("hallucinate")
        if ha != hb:
            diffs.append((_id, ha, hb))

    print(f"\nComparing 'hallucinate':")
    print(f"  A = {file_a}")
    print(f"  B = {file_b}\n")
    if not diffs:
        print("No differences in `hallucinate` field.")
    else:
        print(f"{len(diffs)} differing rows:\n")
        for _id, ha, hb in diffs:
            print(f"  {_id}: A={ha}  ->  B={hb}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "evaluation_v2.jsonl"
    b = sys.argv[2] if len(sys.argv) > 2 else "evaluation_v2_short_eval_prompt_no_question.jsonl"
    main(a, b)

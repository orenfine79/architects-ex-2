"""
Enrich the RAG corpus with synthetic questions: for every chunk in
corpus_chunks.jsonl, ask an LLM to invent 10 questions that chunk can answer,
and write them to a jsonl file. These give us anchor questions to embed for
retrieval eval / hard-negative mining.

    python enrich.py                              # all chunks -> enriched_questions.jsonl
    python enrich.py --limit 20                   # smoke test on 20 chunks
    python enrich.py --model Qwen/Qwen3-32B --workers 16

Re-runnable: chunks whose questions are already in the output file are
skipped, so an interrupted run just picks up where it left off.
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tf_client import chat

MODEL = "google/gemma-3-27b-it"
N_QUESTIONS = 10

EXAMPLE_QUESTIONS = """
הגשתי תביעה לחברת הביטוח והיא עדיין בטיפול כבר הרבה זמן. האם עצם הפנייה לחברה עוצרת את מרוץ ההתיישנות של התביעה שלי?
קרה נזק בדירה שלי לפני חודש. האם נכון שיש לי שלוש שנים מיום המקרה להגיש תביעה לתגמולי ביטוח?
אם תהיה רעידת אדמה והדירה שלי תיפגע, המדינה לא אמורה לפצות אותי בכל מקרה? למה בכלל להוסיף כיסוי כזה לביטוח הדירה?
יש לי ביטוח למחזיקי אקדח ברישיון. אם המבטח לא העמיד לי עורך דין ופניתי לעורך דין לפי בחירתי להגנה בהליך הפלילי, עד איזה סכום ישפו אותי על שכר הטרחה וההוצאות?
אני חבר ועד בית ורוצה לעשות ביטוח לבניין המשותף, וחוץ מזה יש לי אוסף אומנות פרטי שאני רוצה לבטח גם. איך מצטרפים לשני הביטוחים האלה?
הדירה שלי מבוטחת בסכום של 1,200,000 ש"ח. שריפה פגעה בגינה — בדשא, בשיחים ובמערכת ההשקיה. מה הסכום המקסימלי שהביטוח ישלם על הנזק הזה?
אני מעסיק במשרד עובדים דרך קבלן משנה. אם עובד של קבלן המשנה ייפגע ויתבע אותי, האם ביטוח החבות בפוליסת המכלול למשרד יכסה את זה?
יש לי ביטוח ציוד אלקטרוני לעסק, ובאמצע תקופת הביטוח חל שינוי מהותי בעסק שלי. האם אני חייב להודיע על כך לחברת הביטוח בכתב?
אם המשרד שלי יושבת בעקבות נזק מכוסה, לכמה זמן לכל היותר אקבל פיצוי על אובדן ההכנסות?
אני חוקר שמתכנן ניסוי קליני שאושר על ידי ועדת הלסינקי. האם אני חייב לרכוש ביטוח לניסוי?
אחרי נזק בעסק הוצאתי הוצאות נוספות כדי לא לאבד הכנסות. אני משווה בין פוליסת שיר לעסק לבין הראל ביט לאובדן תוצאתי — האם יש תקרה להחזר על הוצאות כאלה בכל אחת מהן?
ביטחתי את התכולה שלי בהראל בסכום ביטוח של 300,000 ש"ח, ובלי לשים לב יש לי גם ביטוח אצל מבטח אחר על אותה תכולה בסכום של 100,000 ש"ח. אם יקרה נזק, איך יתחלק התשלום בין שתי החברות?
המפתחות שלי ננעלו בתוך הרכב – האם מגיע לי שירות פריצה לרכב במסגרת כתב השירות?
יש לי תקר בגלגל וצריך החלפה – האם השירות ניתן בחינם בלי שום תשלום מצדי?
אם אאבד את המפתחות או השלט של הרכב, ביטוח צד ג' של הראל מכסה את זה? זה עולה תוספת?
אני נוסע עם הרכב שלי לטיול בסיני, איפה ואיך אפשר לרכוש את ביטוח הרכב למטיילים של הראל?
אני נוסע מעט מאוד ורוצה ביטוח מקיף שיחסוך לי כסף – מה ההבדל בין הראל סוויץ' לביטוח המקיף הרגיל, ואיך מצטרפים לכל אחד?
ביטלתי את פוליסת ביטוח החובה שלי אחרי שהייתה בתוקף 10 ימים. דמי הביטוח השנתיים ששילמתי הם 2,000 ₪. כמה כסף יחזירו לי (בלי הצמדה למדד)?
סיימתי טיפול שיניים ואני רוצה להגיש תביעה להחזר – אפשר לשלוח לכם את הטופס והמסמכים במייל?
קיבלתי כבר החזר מקופת החולים על טיפול השיניים – מותר לי לתבוע גם מכם החזר על אותה חשבונית?
איך אני יכול לבטל את פוליסת ביטוח השיניים הפרטית שלי בהראל?
הבת שלי בת 17 עברה טיפול שיניים. אפשר להעביר לה את החזר התביעה ישירות לביט שלה?
אני רוצה לבטל את ביטוח השיניים שלי אבל יש לי גם תביעה שעדיין מתבררת. לאיזה מייל שולחים את טופס הביטול, ומה מספר הטלפון של אגף ביטוח השיניים לבירור התביעה?
בכל אחת משלוש הביקורות שעשיתי אצל רופא השיניים השנה ביצעו לי צילום נשך. כמה צילומים יצאו לי בסך הכל?
הראל שלחה לי הצעת פשרה בכתב על התביעה שלי אבל עוד לא אישרתי אותה — האם ההצעה כבר מחייבת אותי?
יש לי כתב שירות לרפואה משלימה וקיבלתי הפניה מרופא — האם טיפולי אקופונקטורה (דיקור סיני) כלולים בשירות?
אני מתלבט בין חברות ביטוח - מה היתרונות של ביטוח הבריאות של הראל לעומת אחרים?
אני סובל מאקנה בעור הפנים. האם אני יכול לקבל שירות עבור זה במסגרת כתב השירות רופא מלווה אישי בתחום מחלות העור?
אמא שלי היא זו שמשלמת על פוליסת הבריאות שלי. אם אחליט לבטל את כתב השירות לרפואה משלימה באמצע התקופה, אקבל החזר? ולאן הכסף יוחזר?
אני בן 35 ורוצה לרכוש את ביטוח הניתוחים UPGRADE משלים שב"ן. כמה יעלה לי הביטוח לשנה שלמה, בלי הנחות?
הפסקתי לעשן לפני יותר משנתיים. האם אני יכול לפנות לחברה כדי שתבחן שינוי בתעריף שלי?
שלחתי בקשה להקטין את סכום הביטוח בפוליסה שלי. האם השינוי יכול להיכנס לתוקף רטרואקטיבית, מחודש שקדם לבקשה?
אם אחליט לבטל את פוליסת ביטוח החיים מגן שלי אחרי כמה שנים, האם אקבל בחזרה חלק מהכסף כערך פדיון?
אני מגיש בקשה להגדלת כיסוי בביטוח החיים שלי. מה יקרה אם לא אצרף טופס הצהרת בריאות לבקשה?
אני באמצע תהליך הצטרפות לביטוח חיים בהראל. לאן אני שולח את הצהרת הבריאות שמילאתי, וממתי הביטוח שלי בכלל נכנס לתוקף?
לא מיניתי מוטבים בפוליסת ביטוח החיים שלי. אם אלך לעולמי, למי ישולם סכום הביטוח, ומי בדיוק נחשב "יורש חוקי"?
הדירה שלי עומדת ריקה בתקופה הקרובה — האם ההרחבה לנזקי מים תכסה נזק שיקרה כשהדירה לא תפוסה?
יש לי בבית דוד שמש גדול של 300 ליטר — אם רכשתי את הכיסוי לדודים, האם נזק לדוד כזה מכוסה?
אם חלילה תפרוץ מלחמה וייגרם נזק לדירה שלי, ביטוח המבנה של המשכנתא יכסה את הנזק?
שמעתי שיש נספח משכנתא מיוחד לאנשים עם מוגבלות מקצרת חיים. מי בכלל נחשב לאדם עם מוגבלות מקצרת חיים לצורך הנספח הזה?
אילו ביטוחי משכנתא בכלל יש אצלכם בהראל, ואם אני רוצה את ביטוח החיים למשכנתא - איך מצטרפים אליו?
הדירה שלי בבית משותף מבוטחת במבנה בסכום של 1,800,000 ₪ ויש לי גם כיסוי סכום נוסף. אחרי רעידת אדמה, שמאי מקרקעין העריך את הנזק לדירה ב-1,350,000 ₪. האם מגיע לי תשלום הסכום הנוסף?
אושפזתי בבית חולים בחו"ל ליותר משלושה ימים. האם הביטוח ישלם על כרטיס טיסה ולינה עבור מלווה שיגיע אליי?
התגלה לי הריון לראשונה בזמן שהייתי בחו"ל ואני כבר אחרי שבוע 12. האם ההוצאות הרפואיות שלי בגלל ההיריון יכוסו?
אם חלילה אמות בתאונה בחו"ל ויש לי את הרחבת תאונות אישיות בדרכון פרימיום, כמה יקבלו היורשים שלי?
אני בחו"ל וגיליתי רק עכשיו, בשבוע 14, שאני בהיריון. האם ההוצאות הרפואיות שלי בקשר להיריון מכוסות בפוליסת דרכון פרימיום?
אם בטעות אגרום נזק למישהו או לרכוש שלו בזמן הטיול, הפוליסה של דרכון first class מכסה אותי? עד איזה סכום, וכמה עולה התוספת הזו?
אני טס לחו"ל עם רחפן יקר. כמה יעלה לי לבטח אותו בנפרד ליום, ומה אצטרך להמציא לחברת הביטוח אם הוא ייגנב לי בחו"ל?
"""

PROMPT = """You are given a single passage from a Harel Insurance document.
Generate exactly {n} distinct questions that this passage can answer. Write each 
question in the same language as the passage. Make them natural questions a real 
customer or agent might ask; vary phrasing and scope. Do NOT answer them.

Here are some examples of questions that were generated either for other passages,
and maybe for this passage. don't reuse the examples as they are.
Use this as an inspiration for the style and scope of the questions you should write:
{example_questions}

Return ONLY a JSON array of {n} strings, nothing else.

PASSAGE:
{text}
"""


def parse_questions(reply):
    """Pull a JSON list of question strings out of the model reply, tolerating
    code fences and stray prose around it."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", reply).strip()
    # fall back to the first [...] block if the model wrapped it in prose
    if not reply.startswith("["):
        m = re.search(r"\[.*\]", reply, re.DOTALL)
        if m:
            reply = m.group(0)
    try:
        data = json.loads(reply)
        return [str(q).strip() for q in data if str(q).strip()]
    except json.JSONDecodeError:
        # The model emits one string per line but often leaves double quotes
        # inside a question unescaped -- Hebrew gershayim like ש"ח, שב"ן, דו"ח --
        # which breaks strict JSON. Recover each line's quoted string greedily so
        # the inner quotes are preserved.
        out = []
        for line in reply.splitlines():
            m = re.match(r'\s*"(.*)"\s*,?\s*$', line)
            if m and m.group(1).strip():
                out.append(m.group(1).strip())
        if out:
            return out
        raise


def gen_for_chunk(chunk, model):
    """Return (chunk_id, [question_text, ...]) for one chunk."""
    prompt = PROMPT.format(n=N_QUESTIONS, example_questions=EXAMPLE_QUESTIONS, text=chunk["text"])
    reply = chat([{"role": "user", "content": prompt}], model=model, quiet=True)
    return chunk["id"], parse_questions(reply)


def load_done(path):
    """chunk_ids already present in an existing output file (for resume)."""
    done = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["chunk_id"])
    except FileNotFoundError:
        pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="corpus_chunks.jsonl")
    ap.add_argument("--out", default="enriched_questions.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only process the first N (new) chunks")
    args = ap.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    done = load_done(args.out)
    todo = [c for c in chunks if c["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(chunks)} chunks, {len(done)} already done, {len(todo)} to process",
          file=sys.stderr)

    lock = threading.Lock()
    written = errors = 0
    with open(args.out, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(gen_for_chunk, c, args.model): c["id"] for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            chunk_id = futures[fut]
            try:
                _, questions = fut.result()
            except Exception as e:  # keep going; a few chunks may fail
                errors += 1
                print(f"[error] {chunk_id}: {e}", file=sys.stderr)
                continue
            with lock:
                for j, q in enumerate(questions):
                    rec = {"question_id": f"{chunk_id}#q{j}",
                           "chunk_id": chunk_id,
                           "question_text": q}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += len(questions)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} chunks, {written} questions, {errors} errors",
                      file=sys.stderr)

    print(f"done: wrote {written} questions ({errors} chunk errors) -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

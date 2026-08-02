#!/usr/bin/env python3
"""
clean_data.py — clean the parsed_corpus JSON documents.

parsed_corpus mirrors the raw corpus but stores each document as a single JSON
file: parsed_corpus/<section>/{pages,files}/<name>.json, shaped as

    {"file": ..., "domain": ..., "kind": "txt"|"pdf", "url": ...,
     "pages": [{"page": 1, "text": "..."}, ...]}

The document's text lives in the per-page ``text`` fields. This script offers
two independent operations:

1. Boilerplate stripping (default) — the scraped web pages (kind == "txt") carry
   the Harel site's global nav + footer on every page, ~69% of each page and
   repeated verbatim. Detection is data-driven: shingle every page into
   overlapping n-grams, count how many pages each n-gram appears in, and strip
   any n-gram present in >= a threshold fraction of pages. Boilerplate is
   detected per directory (i.e. per section's pages/), and the cleaned text is
   written back into each page's ``text`` field in place. PDFs are left alone —
   they don't carry the web chrome.

2. Non-Hebrew deletion (--delete-non-hebrew) — some documents are foreign-language
   translations (English/Arabic/Russian/French policy versions). Any JSON whose
   text is below a Hebrew-character threshold is deleted outright.

Usage:
    python clean_data.py                          # strip boilerplate in parsed_corpus
    python clean_data.py --dry-run                # preview, write nothing
    python clean_data.py --dir parsed_corpus/apartment/pages
    python clean_data.py --delete-non-hebrew            # report non-Hebrew docs
    python clean_data.py --delete-non-hebrew --apply    # actually delete them
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

DEFAULT_DIR = "parsed_corpus"

# HTML entities (&quot; &#x27; …) and non-letters carry no language signal.
_ENTITY = re.compile(r"&[#0-9a-zA-Z]+;")


# --------------------------------------------------------------------------- #
# JSON document I/O
# --------------------------------------------------------------------------- #
def load_doc(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_doc(path: str, doc: dict) -> None:
    # Match the corpus's compact on-disk style (default separators, unicode kept).
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)


def doc_text(doc: dict) -> str:
    """All of a document's text, concatenated across its pages."""
    return " ".join(pg.get("text", "") for pg in doc.get("pages", []))


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def hebrew_fraction(text: str) -> float:
    """Share of alphabetic characters that are Hebrew (U+0590–U+05FF).

    HTML entities are stripped first. Returns 0.0 when the text has no letters
    at all (so empty / punctuation-only docs count as non-Hebrew).
    """
    text = _ENTITY.sub(" ", text)
    hebrew = latin = 0
    for ch in text:
        if "֐" <= ch <= "׿":
            hebrew += 1
        elif ch.isalpha() and ch.isascii():
            latin += 1
    letters = hebrew + latin
    return hebrew / letters if letters else 0.0


# --------------------------------------------------------------------------- #
# Boilerplate detection & stripping
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return text.split()


def shingles(tokens: list[str], n: int):
    """Yield (start_index, ngram_tuple) for every length-n window."""
    for i in range(len(tokens) - n + 1):
        yield i, tuple(tokens[i:i + n])


def build_boilerplate_shingles(token_lists, n: int, min_df_frac: float) -> set[tuple]:
    """Shingles whose document frequency >= min_df_frac * num_docs."""
    token_lists = list(token_lists)
    df = Counter()
    for tokens in token_lists:
        df.update({sh for _, sh in shingles(tokens, n)})
    min_df = max(2, round(min_df_frac * len(token_lists)))
    return {sh for sh, c in df.items() if c >= min_df}


def clean_tokens(tokens: list[str], boiler: set[tuple], n: int) -> list[str]:
    """Drop every position covered by a boilerplate shingle; keep the rest."""
    boiler_pos = [False] * len(tokens)
    for i, sh in shingles(tokens, n):
        if sh in boiler:
            for j in range(i, i + n):
                boiler_pos[j] = True
    return [t for t, is_boiler in zip(tokens, boiler_pos) if not is_boiler]


def clean_boilerplate(directory: str, n: int, min_df_frac: float, dry_run: bool) -> None:
    """Strip shared boilerplate from every page (kind == "txt") JSON under dir.

    Boilerplate is detected independently within each directory group (a
    section's pages/), then removed from each page's ``text`` field in place.
    """
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True))
    docs = {p: load_doc(p) for p in paths}
    page_paths = [p for p in paths if docs[p].get("kind") == "txt"]
    if not page_paths:
        raise SystemExit(f"No page (kind='txt') JSON documents found under {directory!r}")

    groups: dict[str, list[str]] = defaultdict(list)
    for p in page_paths:
        groups[os.path.dirname(p)].append(p)

    print(f"{len(page_paths)} page docs in {len(groups)} group(s) "
          f"| {n}-grams, df >= {min_df_frac:.0%}\n")
    print(f"{'file':<40}{'before':>8}{'after':>8}{'kept':>7}")
    print("-" * 63)
    tot_before = tot_after = 0
    for group_dir in sorted(groups):
        group = sorted(groups[group_dir])
        token_lists = {p: tokenize(doc_text(docs[p])) for p in group}
        boiler = build_boilerplate_shingles(token_lists.values(), n, min_df_frac)
        for p in group:
            doc = docs[p]
            before = after = 0
            for pg in doc.get("pages", []):
                toks = tokenize(pg.get("text", ""))
                kept = clean_tokens(toks, boiler, n)
                before += len(toks)
                after += len(kept)
                pg["text"] = " ".join(kept)
            tot_before += before
            tot_after += after
            pct = (after / before * 100) if before else 0
            print(f"{os.path.basename(p):<40}{before:>8}{after:>8}{pct:>6.0f}%")
            if not dry_run:
                save_doc(p, doc)

    print("-" * 63)
    kept_pct = (tot_after / tot_before * 100) if tot_before else 0
    print(f"{'TOTAL':<40}{tot_before:>8}{tot_after:>8}{kept_pct:>6.0f}%")
    print(f"\nRemoved {tot_before - tot_after} boilerplate tokens "
          f"({100 - kept_pct:.0f}% of page text).")
    print("(dry-run — no files written)" if dry_run else "Documents cleaned in place.")


# --------------------------------------------------------------------------- #
# Non-Hebrew deletion
# --------------------------------------------------------------------------- #
def find_non_hebrew(directory: str, thresh: float) -> list[tuple[str, float]]:
    """Return (json_path, hebrew_fraction) for documents below the threshold."""
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        frac = hebrew_fraction(doc_text(load_doc(p)))
        if frac < thresh:
            out.append((p, frac))
    return out


def delete_non_hebrew(directory: str, thresh: float, apply: bool) -> None:
    """Report (and, when apply=True, delete) JSON documents not in Hebrew."""
    candidates = find_non_hebrew(directory, thresh)
    print(f"Scanned {directory} | Hebrew threshold {thresh:.0%}")
    if not candidates:
        print("No non-Hebrew documents found.")
        return

    print(f"\n{len(candidates)} non-Hebrew document(s):")
    print(f"{'file':<62}{'hebrew%':>8}")
    print("-" * 70)
    for p, frac in sorted(candidates, key=lambda x: x[1]):
        print(f"{os.path.relpath(p):<62}{100 * frac:>7.0f}%")

    if not apply:
        print("\n(report only — pass --apply to delete these documents)")
        return

    for p, _ in candidates:
        os.remove(p)
    print(f"\nDeleted {len(candidates)} non-Hebrew document(s).")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help=f"parsed_corpus directory to process (default: {DEFAULT_DIR})")
    ap.add_argument("--n", type=int, default=8,
                    help="shingle size in tokens (default: 8)")
    ap.add_argument("--min-df-frac", type=float, default=0.4,
                    help="a shingle is boilerplate if it appears in >= this "
                         "fraction of pages (default: 0.4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report but write/delete nothing")
    ap.add_argument("--delete-non-hebrew", action="store_true",
                    help="instead of stripping boilerplate, find documents that "
                         "are not in Hebrew and delete them (reports only unless "
                         "--apply is given)")
    ap.add_argument("--hebrew-thresh", type=float, default=0.25,
                    help="min share of letters that must be Hebrew to keep a "
                         "document (default: 0.25)")
    ap.add_argument("--apply", action="store_true",
                    help="with --delete-non-hebrew, actually delete the files")
    args = ap.parse_args()

    if args.delete_non_hebrew:
        delete_non_hebrew(args.dir, args.hebrew_thresh,
                          apply=args.apply and not args.dry_run)
        return

    clean_boilerplate(args.dir, args.n, args.min_df_frac, args.dry_run)


if __name__ == "__main__":
    main()

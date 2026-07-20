"""
Stage 2 RAG: run the dev questions straight through a bare model WITH
retrieval, producing an answers file the eval harness can score.

    # set OPENAI_BASE_URL and OPENAI_API_KEY in .env file
    uv run rag_runner.py
    # then score rag_answers.jsonl with YOUR evaluation harness (Stage 2)

Calls go through litellm: a bare model name goes to OpenAI; set
OPENAI_BASE_URL for any OpenAI-compatible endpoint (Token Factory, a local
vLLM server, ...); provider-prefixed models ("anthropic/...", "gemini/...")
work with the matching key env var. Try --system-prompt variants and watch
how the failure profile (not just the score) changes.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

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


CORPUS_DIR = Path("corpus")
PARSED_DIR = Path("parsed_corpus")


def _make_converter():
    # imported lazily: docling drags in torch and layout models, which only
    # the parsing path needs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # the corpus PDFs carry a text layer; OCR would need Hebrew models
    opts.do_table_structure = True  # coverage/deductible tables must survive
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse_pdf(converter, path: Path) -> list[dict]:
    doc = converter.convert(path).document
    pages = [{"page": n, "text": doc.export_to_text(page_no=n)}
             for n in sorted(doc.pages)]
    if not pages:  # rare: docling found no page structure, keep whatever text exists
        text = doc.export_to_text()
        pages = [{"page": 1, "text": text}] if text.strip() else []
    return pages


def parse_page_txt(path: Path) -> list[dict]:
    return [{"page": 1, "text": path.read_text(encoding="utf-8")}]


def parse_corpus(corpus_dir: Path = CORPUS_DIR, parsed_dir: Path = PARSED_DIR,
                 force: bool = False) -> list[dict]:
    """Parse (or load from cache) every corpus document. Returns the parsed docs.
    With force=True, cached results are ignored and every document is re-parsed."""
    manifest_file = corpus_dir / "manifest.json"
    manifest = (json.loads(manifest_file.read_text(encoding="utf-8"))
                if manifest_file.exists() else {})

    sources = sorted(corpus_dir.glob("*/pages/*.txt")) + sorted(corpus_dir.glob("*/files/*.pdf"))
    converter = None
    docs, failed = [], []
    for i, src in enumerate(sources, 1):
        rel = src.relative_to(corpus_dir).as_posix()
        cache = parsed_dir / f"{rel}.json"
        if cache.exists() and not force:
            rec = json.loads(cache.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            try:
                if src.suffix == ".pdf":
                    converter = converter or _make_converter()
                    pages = parse_pdf(converter, src)
                else:
                    pages = parse_page_txt(src)
                rec = {"file": rel, "domain": rel.split("/")[0],
                       "kind": src.suffix.lstrip("."), "url": manifest.get(rel),
                       "pages": pages}
                print(f"[{i}/{len(sources)}] {rel}: {len(pages)} pages "
                      f"({time.time() - t0:.1f}s)")
            except Exception as e:
                rec = {"file": rel, "error": f"{type(e).__name__}: {e}"}
                print(f"[{i}/{len(sources)}] {rel}: FAILED ({rec['error']})")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        (failed if "error" in rec else docs).append(rec)

    if failed:
        print(f"\nWARNING: {len(failed)} documents failed to parse "
              f"(cached as errors -- delete their .json under {parsed_dir}/ to retry):")
        for rec in failed:
            print(f"  {rec['file']}: {rec['error']}")
    return docs


CHUNKS_FILE = Path("corpus_chunks.jsonl")
DEFAULT_CHUNK_SIZE = 1500     # chars; ~350-450 tokens of Hebrew, well inside embedding limits
DEFAULT_CHUNK_OVERLAP = 200   # chars of trailing context carried into the next chunk


def _split_oversized(paragraph: str, size: int) -> list[str]:
    """Cut a paragraph longer than `size` at sentence ends (hard cut as last resort)."""
    pieces = []
    while len(paragraph) > size:
        cut = max(paragraph.rfind(end, 0, size) for end in (". ", "? ", "! ", ".\n", ":\n", "\n"))
        if cut < size // 2:  # no usable boundary in the back half -- hard cut
            cut = size - 1
        pieces.append(paragraph[:cut + 1].strip())
        paragraph = paragraph[cut + 1:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Greedily pack paragraphs into ~size-char chunks; whole trailing paragraphs
    up to `overlap` chars are repeated at the start of the next chunk so facts
    straddling a boundary stay retrievable."""
    units = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            units.extend(_split_oversized(para, size) if len(para) > size else [para])

    chunks, cur, cur_len = [], [], 0
    for unit in units:
        if cur and cur_len + len(unit) > size:
            chunks.append("\n\n".join(cur))
            tail, tail_len = [], 0
            for prev in reversed(cur):  # seed the next chunk with the overlap tail
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            if not tail and overlap:  # trailing paragraph too long -- carry its last sentences
                t = cur[-1][-overlap:]
                cut = t.find(". ")
                t = t[cut + 2:] if 0 <= cut < overlap // 2 else t[t.find(" ") + 1:]
                tail, tail_len = [t], len(t)
            cur, cur_len = tail, tail_len
        cur.append(unit)
        cur_len += len(unit) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_corpus(docs: list[dict], out_path: Path = CHUNKS_FILE,
                 size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    """Chunk parsed docs into vector-DB-ready records and write them to out_path.

    Chunks never cross page boundaries, so every chunk maps to one citable
    {file, page}. Each record: {"id", "file", "domain", "kind", "url", "page",
    "chunk_index", "text"}.
    """
    chunks = []
    for doc in docs:
        for page in doc["pages"]:
            for j, text in enumerate(chunk_text(page["text"], size, overlap)):
                chunks.append({"id": f"{doc['file']}#p{page['page']}.{j}",
                               "file": doc["file"], "domain": doc["domain"],
                               "kind": doc["kind"], "url": doc.get("url"),
                               "page": page["page"], "chunk_index": j,
                               "text": text})
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sizes = sorted(len(c["text"]) for c in chunks)
    print(f"chunked {len(docs)} docs -> {len(chunks)} chunks -> {out_path} "
          f"(chars/chunk: median {sizes[len(sizes) // 2]}, max {sizes[-1]})")
    return chunks


CHUNK_EMBEDDINGS_FILE = Path("chunk_embeddings.npz")
QUESTION_EMBEDDINGS_FILE = Path("question_embeddings.npz")
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"  # strong multilingual/Hebrew retriever
DEFAULT_BATCH_SIZE = 32


def _load_embedder(model_name: str):
    # imported lazily: sentence-transformers pulls in torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _e5_prefix(model_name: str, kind: str) -> str:
    # E5-family models are trained with "query: " / "passage: " prefixes;
    # skipping them costs real retrieval quality
    return f"{kind}: " if "e5" in model_name.lower() else ""


def embed_texts(embedder, model_name: str, texts: list[str], kind: str = "passage",
                batch_size: int = DEFAULT_BATCH_SIZE):
    prefix = _e5_prefix(model_name, kind)
    return embedder.encode([prefix + t for t in texts], batch_size=batch_size,
                           normalize_embeddings=True,  # unit vectors: cosine == dot product
                           show_progress_bar=len(texts) > batch_size)


def _embedding_fingerprint(model_name: str, kind: str, items: list[tuple[str, str]]) -> str:
    """Hash of everything the embeddings depend on: model, prefix convention,
    and every (id, text) pair. Changing any of them invalidates caches."""
    import hashlib

    h = hashlib.sha256(model_name.encode())
    h.update(_e5_prefix(model_name, kind).encode())
    for id_, text in items:
        h.update(id_.encode())
        h.update(text.encode())
    return h.hexdigest()


def _embed_cached(items: list[tuple[str, str]], kind: str, out_path: Path, label: str,
                  model_name: str = DEFAULT_EMBED_MODEL,
                  batch_size: int = DEFAULT_BATCH_SIZE, force: bool = False):
    """Embed (id, text) items, cached in out_path keyed by _embedding_fingerprint.
    With force=True, the cache is ignored and everything is re-embedded."""
    import numpy as np

    fingerprint = _embedding_fingerprint(model_name, kind, items)

    if out_path.exists() and not force:
        cached = np.load(out_path)
        if str(cached["fingerprint"]) == fingerprint:
            print(f"{label} embeddings: cache hit ({out_path}, {cached['embeddings'].shape})")
            return cached["embeddings"]

    print(f"embedding {len(items)} {label}s with {model_name} ...")
    t0 = time.time()
    embedder = _load_embedder(model_name)
    embeddings = embed_texts(embedder, model_name, [text for _, text in items],
                             kind=kind, batch_size=batch_size)
    np.savez_compressed(out_path, embeddings=embeddings,
                        ids=np.array([id_ for id_, _ in items]),
                        fingerprint=np.array(fingerprint))
    print(f"embedded {len(items)} {label}s -> {out_path} "
          f"(shape {embeddings.shape}, {time.time() - t0:.0f}s)")
    return embeddings


def embed_chunks(chunks: list[dict], model_name: str = DEFAULT_EMBED_MODEL,
                 out_path: Path = CHUNK_EMBEDDINGS_FILE,
                 batch_size: int = DEFAULT_BATCH_SIZE, force: bool = False):
    return _embed_cached([(c["id"], c["text"]) for c in chunks], kind="passage",
                         out_path=out_path, label="chunk",
                         model_name=model_name, batch_size=batch_size, force=force)


def embed_questions(questions: list[dict], model_name: str = DEFAULT_EMBED_MODEL,
                    out_path: Path = QUESTION_EMBEDDINGS_FILE,
                    batch_size: int = DEFAULT_BATCH_SIZE, force: bool = False):
    return _embed_cached([(q["id"], q["question"]) for q in questions], kind="query",
                         out_path=out_path, label="question",
                         model_name=model_name, batch_size=batch_size, force=force)


CHROMA_COLLECTION = "harel_corpus"


def build_vector_db(chunks: list[dict], embeddings, name: str = CHROMA_COLLECTION):
    """Insert chunks + precomputed embeddings into an in-memory chromadb
    collection, rebuilt on every run (the embeddings themselves stay cached
    on disk, so this only costs a few seconds)."""
    import chromadb

    t0 = time.time()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    batch_size = 2048
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(ids=[c["id"] for c in batch],
                embeddings=embeddings[i:i + batch_size],
                documents=[c["text"] for c in batch],
                metadatas=[{"file": c["file"], "page": c["page"],
                            "domain": c["domain"], "kind": c["kind"],
                            "url": c.get("url") or "", "chunk_index": c["chunk_index"]}
                           for c in batch])
    print(f"vector db: inserted {collection.count()} chunks into in-memory collection "
          f"'{name}' ({time.time() - t0:.0f}s)")
    return collection


DEFAULT_TOP_K = 5


def query_top_k(collection, query_embedding, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the k nearest chunks as dicts: metadata + text + cosine score."""
    res = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [{**meta, "text": doc, "score": 1 - dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                       res["distances"][0])]


def build_rag_prompt(question: str, hits: list[dict]) -> str:
    """User message: numbered source chunks + the question + citation protocol."""
    sources = "\n\n---\n\n".join(
        f"[{i}] file: {h['file']} | page: {h['page']}\n{h['text']}"
        for i, h in enumerate(hits, 1))
    return f"""Below are excerpts from official Harel insurance documents, followed by a customer question.

SOURCES:
{sources}

---

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the sources above; do not use outside knowledge.
- If the sources do not contain enough information to answer, say you do not have the exact information.
- End your reply with one final line listing the numbers of the sources that support your answer, exactly in this format: CITED: 1,3
- If no source supports the answer, end with: CITED: none"""


_CITED_RE = re.compile(r"CITED:\s*(none|[\d,\s]+)\s*$", re.IGNORECASE)


def extract_citations(answer: str, hits: list[dict]) -> tuple[str, list[dict]]:
    """Split the CITED: trailer off the model reply and resolve the numbers to
    {file, page} citations (page is null for web pages, per the contract)."""
    m = _CITED_RE.search(answer.strip())
    if not m:
        return answer.strip(), []
    citations, seen = [], set()
    for tok in re.findall(r"\d+", m.group(1)):
        i = int(tok) - 1
        if 0 <= i < len(hits):
            h = hits[i]
            page = h["page"] if h["kind"] == "pdf" else None
            if (h["file"], page) not in seen:
                seen.add((h["file"], page))
                citations.append({"file": h["file"], "page": page})
    return answer.strip()[:m.start()].rstrip(), citations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--system-prompt", default=V4_SYSTEM)
    ap.add_argument("--out", default="rag_answers.jsonl")
    ap.add_argument("--eval", action="store_true",
                    help="after generating answers, score them with the eval harness")
    ap.add_argument("--eval-out", default="evaluation.jsonl",
                    help="where to write per-item eval results (with --eval)")
    ap.add_argument("--parse-corpus", action="store_true",
                    help="parse the corpus into parsed_corpus/ and exit")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="target chunk size in characters")
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help="characters of trailing context repeated in the next chunk")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="sentence-transformers model for chunk/query embeddings")
    ap.add_argument("--embed-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="encode batch size for embedding")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="number of chunks to retrieve per question")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached files: re-parse the corpus "
                         "and re-embed chunks and questions")
    args = ap.parse_args()

    # loads from parsed_corpus/ cache; parses (docling) anything missing
    docs = parse_corpus(force=args.force)
    n_pages = sum(len(d["pages"]) for d in docs)
    print(f"corpus: {len(docs)} documents / {n_pages} pages (cache: {PARSED_DIR}/)")
    if args.parse_corpus:
        return

    chunks = chunk_corpus(docs, size=args.chunk_size, overlap=args.chunk_overlap)
    embeddings = embed_chunks(chunks, model_name=args.embed_model,
                              batch_size=args.embed_batch_size, force=args.force)
    collection = build_vector_db(chunks, embeddings)

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

    # embed all questions (cached on disk), then query the db per question
    q_vecs = embed_questions(questions, model_name=args.embed_model,
                             batch_size=args.embed_batch_size, force=args.force)

    with open(args.out, "w", encoding="utf-8") as out:
        for q, q_vec in zip(questions, q_vecs):
            t0 = time.time()
            hits = query_top_k(collection, q_vec, k=args.top_k)
            resp = litellm.completion(model=model, messages=[
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": build_rag_prompt(q["question"], hits)}],
                timeout=120, **kwargs)
            answer, citations = extract_citations(resp.choices[0].message.content, hits)
            rec = {"id": q["id"],
                   "answer": answer,
                   "citations": citations,
                   "retrieved": [{"file": h["file"], "page": h["page"],
                                  "score": round(h["score"], 3)} for h in hits],
                   "latency_ms": (time.time() - t0) * 1000,
                   "tokens": {"prompt": resp.usage.prompt_tokens,
                              "completion": resp.usage.completion_tokens}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {rec['answer'][:70]!r}... ({rec['latency_ms']:.0f} ms; "
                  f"top hit {hits[0]['file']} p{hits[0]['page']} @{hits[0]['score']:.2f})")
    
    print(f"\nwrote {args.out} -- now score it with your evaluation harness")

    if args.eval:
        print("\nscoring answers with the eval harness ...")
        run_evaluation(questions, args.out, args.eval_out)


if __name__ == "__main__":
    main()

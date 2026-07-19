# Exercise 2 — Build a Domain-Specific Customer Support Chatbot for Harel Insurance

**Timeline:** [DATE TBD - will be given in class] · **Teams:** 4–5 participants · **Mentorship:** mentors join the later sessions to assist and brainstorm with teams · **Final presentations:** [DATE TBD - will be given in class]

---

## 🚀 Overview

This capstone simulates a real-world, high-stakes AI systems challenge: building a production-grade, domain-specific customer support chatbot for Israel's largest insurance provider.

You will design and implement an end-to-end GenAI system that:

- **Ingests and structures** real insurance policy data
- **Answers customer questions** across twelve insurance domains (Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment, Long-Term Care, Personal Accident, Diseases & Disabilities, Loss of Working Ability)
- **Grounds every answer** in official documentation, with explicit citations
- **Outperforms a bare LLM baseline**
- **Works with open-weights models** served via Nebius Token Factory

This is not a toy demo. The final deliverable should resemble a system that could realistically power an insurer's first-line support chatbot.

**Why this challenge matters.** Real-world AI systems fail not because models are weak, but because:

- Data is messy and unstructured
- Knowledge must be grounded and verifiable
- Evaluation is subtle and unforgiving

Modern models are extremely strong — they will answer many insurance questions plausibly from memory alone. That is exactly the trap: *plausible* is not *grounded*, and in insurance, an ungrounded answer is a lawsuit. Your job is to build the system that knows what it knows.

---

## 🧠 The Challenge — three stages

### Stage 1: Baseline & Evaluation Harness

**Goal:** establish a strong baseline and learn to measure progress.

You will:

1. Measure the baseline: Run the provided development question set (`reference_questions.json`) through the **strongest open-weights model on Nebius Token Factory** (e.g. the latest DeepSeek or GLM model) as a bare model — just the question, no documents. Use the provided `baseline_runner.py` with `OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1`. **All teams share one course API key** — make your calls through the provided `tf_client.py`, watch the cost estimate it prints, and play fair with the shared balance.
2. **Build an evaluation harness** — It must score a batch of answers against the dev set's ground truths and report at least:
   - **Answer relevance** — does the answer agree with the ground-truth answer on the asked fact? Use an LLM-as-judge (a Token Factory model works); force structured JSON output and pin the judge model so runs are comparable.
   - **Hallucination rate** — confident answers that *contradict* the ground truth. Decide how your judge treats refusals ("I don't know" is not a hallucination — a system that knows what it doesn't know is worth measuring).
   - **Citation accuracy** — does the cited evidence actually establish the answer? Resolve each cited `{file, page}` to the actual corpus page and use an LLM judge to decide whether the cited pages establish the **ground-truth answer** (fully / partially / not at all). The same fact often appears in several corpus documents — *any* page that truly establishes it earns credit; there is no fixed list of "correct" sources to match against. The `ground_truth_sources` in `reference_questions.json` show where each answer was authored from — useful for debugging your retrieval, but not the scoring target. A citation pointing at a nonexistent file or page counts against you. (For the bare baseline, expect ~zero — the model has nothing to cite.)
   - **Latency** per question.
   Your final grade uses our internal harness (same criteria, plus conversational quality and efficiency), so a harness that tracks these honestly is your compass for the whole exercise. Wire it into your loop from day one.
3. Experiment with at least two prompt strategies (e.g. "answer only if certain", "always cite your source", few-shot-prompting) and observe how the failure profile shifts.

**Questions to answer in your baseline report:**

- Where does the baseline succeed *without* any Harel documents? What does that tell you about what's in its training data?
- When it's wrong, is it wrong *confidently*? Which failure is worse for an insurer — a wrong answer or "I don't know"?
- The judge is itself an LLM. Find one question where you disagree with the judge's verdict. What does that imply about your evaluation at scale?

**Deliverable:** a 1-page baseline report with the metrics table and the three answers above.

### Stage 2: Retrieval Pipeline (RAG Core)

**Goal:** beat the baseline with retrieval + grounding. Same model family, possibly even a *smaller* open-weights model — a model that *reads the right page* should beat a bigger one that *remembers the internet*. Prove it.

**One hard rule: you may NOT fine-tune, LoRA, or RL-train the LLM.** All the intelligence you add lives in the *system around it*.

**The corpus is provided** — we already scraped Harel's official insurance content for all 12 domains (~570 documents: policy PDFs and web pages, mostly Hebrew). Download it with the provided `get_corpus.py` (public dataset: [`orik/apex-ex2-harel-corpus`](https://huggingface.co/datasets/orik/apex-ex2-harel-corpus)). Do not re-scrape the live site — it has drifted from the ground-truth answers. Your work starts at parsing and structuring the corpus.

**1. Build retrieval**

- Parse the documents, chunk them, and build a search index over the corpus. Preserve structure — sections, tables, and **page numbers**; citations require them.
- Return top-k passages with metadata (file, page, domain).

**Questions to guide your design (worth discussing before coding):**

- What's your chunk size, and why? What breaks with page-sized chunks? With sentence-sized ones?
- Embedding-based search has real failure modes on this corpus — find them on the dev set. How would you improve retrieval where embeddings alone fall short?
- How do you *know* your retrieval works, independent of generation?

**2. Generate grounded answers**

- Answer strictly from retrieved context; attach a citation (file + page) to every factual claim.
- Implement a safe fallback when evidence is missing — "I don't have enough information" beats a confident guess.

**Outcome:** a working RAG system, measurably better than the Stage 1 baseline on relevance, hallucination rate, and citation accuracy — with an open-weights generator.

### Stage 3: Agentic Flow & Systemization

**Goal:** build a robust, production-style AI system.

You will:

- Design a single- or multi-agent architecture
- Handle ambiguity and cross-domain questions
- Package the system behind the **provided FastAPI contract** (`contract.py`) — the final evaluation calls your `/ask` endpoint, so the contract is not optional
- We measure you on efficiency as well — you can't optimize what you don't measure

**Optional bonus:** voice interface, simple UI.

**Outcome:** a realistic customer-support AI system with clear separation of concerns, suitable for real deployment.

---

## 📘 Background: About Harel Insurance

Harel Insurance is Israel's largest insurance and financial services group, serving millions of customers across health, life, general insurance, and long-term savings. In this capstone, Harel serves as a realistic enterprise customer with:

- Broad and fragmented product coverage
- Highly regulated, legally precise documentation (Hebrew + some English)
- Complex policy structures that vary by product, customer type, and conditions

Questions range from simple eligibility checks to nuanced policy conditions and exclusions. Accuracy, grounding, and trust are non-negotiable.

**Dev vs. blind evaluation:** you receive a dev question set (`reference_questions.json`) covering a subset of the 12 domains. Every question carries a `difficulty` label — for each covered domain you get 2 **easy** (a yes/no answerable from one document), 2 **medium** (a specific checkable fact — a number, limit, waiting period, exclusion), and 2 **hard** (requires combining several documents and/or a calculation). The final evaluation uses a much larger **hidden blind question set** with the same difficulty structure, covering all 12 domains — including domains with no dev questions. This tests domain generalization, retrieval robustness, and grounding without memorization. Overfitting your pipeline to the dev questions is a strategy — a bad one.

---

## 🏆 The Competition

You are competing against: the **bare-LLM baseline**, the other APEX teams, and — realistically — against how GenAI systems fail in production 🙂

**How submission works:** on submission day you receive `blind_questions.json` — the hidden blind set, questions only. You run the provided `submit_runner.py` on your own machine: it asks your local `/ask` endpoint every question, measures latency, and writes one answers JSONL. You send us that file; we score it with our internal harness. At final presentations we re-ask a few blind questions against your live system — the answers should match your submission.

**How you'll be judged:** answer relevance against the ground-truth answers (LLM-as-judge), citation accuracy (LLM-as-judge: do the pages you cite establish the ground-truth answer?), efficiency (latency and cost), and conversational quality — the same criteria your Stage 1 harness tracks, so your dev-set numbers should roughly predict your blind-set numbers.

---

## 🛠️ Recommended Open-Source Stack

*(Not mandatory, but a reasonable starting point)*

- **Document processing:** Docling
- **Vector DB:** Qdrant / Chroma / Milvus
- **Agent framework:** LangGraph / LangChain / plain Python
- **Evaluation:** your own harness (Stage 1); RAGAS / Opik for extra analysis
- **API:** FastAPI (contract provided)

## 📦 What you get

| File | What it is |
|---|---|
| `get_corpus.py` | Downloads the corpus: ~570 Harel documents (PDFs + web pages), all 12 domains |
| `reference_questions.json` | Dev Q&A set: questions labeled easy/medium/hard, ground-truth answers + reference file/page pointers (where each answer was authored from — for debugging retrieval, not the scoring target) |
| `contract.py` | The FastAPI request/response schema your system must expose |
| `baseline_runner.py` | Stage 1 baseline: questions → bare model → answers JSONL |
| `submit_runner.py` | Batch-asks your `/ask` endpoint and writes the answers JSONL — the exact script used for final submission |
| `tf_client.py` | Minimal Token Factory client with a per-call cost estimate — shared key, play fair |

Everything else — the evaluation harness, parser, chunker, index, retriever, agent — is yours to design.

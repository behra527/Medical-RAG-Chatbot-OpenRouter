# Veritas Health — Premium Medical RAG SaaS-Style Prototype (v2)

A clean, responsive clinical guideline question-answering application built on the existing FastAPI/OpenRouter RAG backend, with a **new dependency-free premium web frontend** served from the FastAPI root URL.

> **Clinical boundary:** This is an educational guideline research application. It is **not a clinically validated medical device**, and it is not ready for unsupervised diagnosis, prescribing or other clinical decisions. Avoid entering identifiable patient information. A qualified clinician must check answers and sources.

## New in v2

- **Fixes “Not Found” at `/`:** `http://127.0.0.1:8000/` now serves the actual application. The older optional Streamlit interface remains at `ui/streamlit_app.py` (install `requirements-streamlit.txt` to run it), but it is not the premium UI.
- **Premium web frontend:** responsive sidebar, evidence-first chat, suggested questions, conversation export, citation cards, original PDF viewer, document library, source search, evaluation dashboard, admin setup, health indicators, proper empty states and helpful/not-helpful feedback.
- **Authenticated administrative actions:** upload, delete, trusted publisher import, evaluation and feedback statistics require `X-Admin-Key` when `ADMIN_API_KEY` is configured. Production mode refuses admin actions without a configured key.
- **Official publisher catalogue:** WHO hypertension 2021 and NICE NG136 2026, with original publisher links, rights notices, and **one-click download + indexing** after authorized admin confirmation.
- **Preserved original PDFs:** local `data/guidelines/<sha256-id>.pdf` files allow citations to open PDF page `#page=N`. The page is the 1-based PDF page number; section headings come from text extraction and must be verified.
- **RAG backbone preserved:** chunking, OpenRouter embeddings, cosine similarity, SQLite FTS5 lexical search, reciprocal-rank fusion, relevance gate, grounding prompt, citation ID verification, retrieval and answer evaluation.

## 1. One-time setup — Windows PowerShell

**Recommended Python:** 3.11 (standard build, not Python 3.13 free-threading). Check installation:

```powershell
py -0p
py -3.11 --version
```

In the **extracted project root** (the folder containing `requirements.txt`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Replace the two placeholders in `.env` with your own secrets:

```dotenv
OPENROUTER_API_KEY=your_real_openrouter_key
ADMIN_API_KEY=a_unique_long_random_secret_only_you_know
```

**Never embed OpenRouter keys in frontend code, post them in chat, or commit `.env` to Git.** The same OpenRouter key is used for the chat and embedding APIs, and both can incur cost. Default embedding model: `openai/text-embedding-3-small`; default chat model: `openai/gpt-4o-mini`.

### Start the whole application (one terminal)

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Or run `start_windows.bat` after installing dependencies and setting `.env`.

- **Premium application:** http://127.0.0.1:8000/
- **Swagger API:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

**You do not need to start Streamlit** for the premium application. The FastAPI service provides both the user interface and backend under one origin.

## 2. Populate the knowledge base with real guideline PDFs

**Important:** This ZIP does not embed full copyrighted clinical guideline books. The runtime catalogue offers verified publisher URLs, and the administrator downloads/indices a PDF **after confirming content rights**. You need internet access and OpenRouter API credits; there is no fake pre-populated corpus or manufactured benchmark.

**Easy method using the web UI:**

1. Open the app at `http://127.0.0.1:8000/` and choose **Knowledge Library**.
2. Under **Administrator access**, enter the same `ADMIN_API_KEY` stored on the backend. Select **Use key for this session**.
3. In the **Official publisher catalogue**, select the WHO hypertension 2021 document. Read the licence notice. Only confirm and import if your use is lawful.
4. The backend downloads the official PDF, extracts pages and sections, creates OpenRouter embeddings, saves its original PDF, and indexes passages into SQLite. **This can incur usage charges.**
5. When indexed, the indicator becomes **Ready to answer**. Ask questions on the **AI Assistant** page. Citations contain verified source identifiers, PDF pages, section labels and supporting excerpts.

**Command-line alternative** (also requires a rights confirmation):

```powershell
python scripts/bootstrap_guidelines.py --guideline who-hypertension-2021 --confirm-rights
```

The NICE option is available **only if you have appropriate rights**:

```powershell
python scripts/bootstrap_guidelines.py --guideline nice-ng136-2026 --confirm-rights
```

You can also upload an authorized PDF in Knowledge Library and add its real title and HTTPS publisher source URL.

### Clinical content rights

- **WHO hypertension guideline (2021):** CC BY-NC-SA 3.0 IGO. Noncommercial use with attribution and share-alike obligations. **Do not assume this licence permits a commercial SaaS offering.** Commercial use needs WHO permission. See [WHO publication](https://www.who.int/publications/i/item/9789240033986).
- **NICE NG136 (last updated February 2026):** UK Open Content Licence is UK-specific; NICE requires permission for many uses outside the UK beyond personal study/research. Do **not** commercially deploy worldwide without rights clearance. See [NICE terms](https://www.nice.org.uk/terms-and-conditions) and [NICE NG136](https://www.nice.org.uk/guidance/ng136).
- The importer uses a fixed publisher URL allowlist. For commercial deployment, secure an appropriate medical guideline corpus licence first.

## 3. Questions to try

For a hypertension guideline corpus, ask:

- How does NICE recommend confirming hypertension for clinic BP between 140/90 and 180/120 mmHg?
- What if ambulatory blood pressure monitoring is unsuitable?
- What blood pressure targets are specified for adults aged 80 and over?
- What drug classes does WHO recommend as initial pharmacological treatment for hypertension?
- What follow-up intervals are recommended for blood pressure reassessment?

**Out-of-scope fallback test:** “What is the standard chemotherapy for leukemia?” The assistant should explicitly abstain if it lacks suitable indexed sources, rather than fabricate citations.

When no documents have been indexed, the UI gives a plain-language **setup required** message rather than claiming that the system is ready. When OpenRouter credentials or the upstream model are unavailable, a clear API error is shown to the user.

## 4. Run automated tests and evaluation

```powershell
python -m pytest -q
python scripts/evaluate.py --retrieval-only
python scripts/evaluate.py
python scripts/evaluate.py --judge
```

The UI's **Evaluation Lab** also runs the included sample questions. The gold question file is `data/eval_nice.json`; this reference set is designed for NICE NG136, so index the correct guideline before running it. Automatic metrics: retrieval Hit@k, MRR@k, answer keyword coverage proxy, abstention, citation presence, latency and optional LLM-as-judge estimates. **No live clinical accuracy figure is claimed until real evaluation is run.**

The tests use synthetic PDFs and deterministic fake OpenRouter clients, so they validate the software pipeline and endpoint contracts—not clinical correctness or live OpenRouter billing/latency.

## 5. Technical architecture

```text
Browser: ui/web/index.html + assets/styles.css + assets/app.js
  | same-origin HTTP/JSON (no npm installation)
FastAPI: app/api.py
  | upload / official catalog import
  v
PyMuPDF -> 1-based PDF page, section detector, chunking + overlap
  | OpenRouter POST /api/v1/embeddings
  v
SQLite persistent vectors + FTS5 BM25 lexical index
  | question -> query embedding -> cosine + lexical reciprocal-rank fusion
  | threshold gate -> context passages
  v
OpenRouter POST /api/v1/chat/completions
  | grounding prompt + whitelisted [S1], [S2], etc
  v
Answer + actual indexed source metadata + original PDF #page=N
```

Important endpoints: `GET /`, `GET /health`, `GET /documents`, `GET /documents/{id}/pdf`, `GET /api/catalog`, `POST /api/catalog/import`, `POST /documents/upload`, `DELETE /documents/{id}`, `POST /chat`, `POST /retrieve`, `GET /api/system`, `POST /api/feedback`, `POST /evaluation/run`.

## 6. Limitations before a public/clinical production launch

This is a functional SaaS-style **engineering prototype**, **not a production-cleared healthcare SaaS**. Before handling real patient data or monetizing, you must add: formally licensed guideline corpus, independent clinical validation, clinical safety escalation policy, robust individual user authentication and RBAC, per-user/organization data separation, secure secret management, HTTPS, rate limiting and abuse prevention, audit logs, monitoring, legal/privacy review, vendor data-processing agreements, retention/deletion policies, and regional regulatory assessment. There is no backend chat-history storage in this version; the browser keeps open conversation state only in memory. Feedback stores only aggregated positive/negative votes, no question text.

Retrieval threshold `MIN_SIMILARITY=0.40` is a **demo starting point, not a calibrated confidence score**. Source-section inference from PDF text can fail for OCR scans, complex tables and columns. Citation IDs confirm only that an indexed passage exists; a generated sentence can still contain an incorrect or unsupported claim. A medically trained reviewer must verify evidence before clinical use.

# Evaluation Status — Veritas Health v2

Prepared 13 November 2026.

## Verified engineering results

- **18/18 automated tests passing**, using synthetic PDF fixtures and deterministic simulated OpenRouter embeddings and completions.
- Covered: input validation, RAG retrieval, document deduplication, citation-ID whitelist, relevance fallback, offline evaluation, FastAPI chatbot, root premium web page, static assets, health readiness, official source catalogue, rights confirmation, original PDF preservation and deletion, feedback, admin authorization, and production-mode admin gating.
- **Live OpenRouter calls not performed.** There was no user-supplied API key in the build environment.
- **No real publisher PDF indexed in the delivered archive.** Access to external publisher PDF download endpoints from the build environment was unavailable. The app has an administrator-controlled one-click official-source downloader that executes on your machine after rights confirmation, internet access and API configuration.

## Real-clinical-dataset metrics — pending

| Metric | Verified live result |
|---|---|
| Clinical retrieval Hit@k | Not measured |
| Clinical retrieval MRR@k | Not measured |
| Real OpenRouter generation latency | Not measured |
| Answer correctness/completeness | Not clinician-reviewed |
| Citation-to-claim entailment | Not independently validated |
| Abstention calibration | Not measured on clinical corpus |

### Reproducible evaluation procedure

1. Configure the OpenRouter API key and ensure billing/quota permits embeddings and chat.
2. Obtain appropriate licensing permission and index a genuine guideline using Knowledge Library or `scripts/bootstrap_guidelines.py`.
3. Confirm that each gold evaluation question points to the correct publisher guideline/section in `data/eval_nice.json`.
4. Run `python scripts/evaluate.py --retrieval-only`, then `python scripts/evaluate.py` and optionally `python scripts/evaluate.py --judge`.
5. Review real generated reports under `data/reports/`. Include model IDs, corpus version, sample size, hardware/network and timing date in any submitted report.
6. For a credible clinical assessment, have clinicians manually grade correctness, limitations, omissions, applicability, evidence support, safety and version conflicts.

### Known engineering limitations

- The PDF section detector is heuristic and may mislabel some headings. 1-based PDF page numbering is preserved even when printed page labels differ.
- Some PDFs contain non-selectable scans, complex tables or embedded fonts requiring specialist OCR or structured parsing.
- WHO content has a noncommercial licence; many NICE international uses require explicit permission. Do not package or monetize publisher PDFs without valid rights.
- Similarity threshold and FTS5/semantic rank fusion should be tuned with a larger diverse test dataset and hard negative examples, including adversarial prompts.
- Citation whitelist checks ID existence, not whether the answer's every medical claim is supported by that source.
- The UI is designed for usability but not accessibility-audited across assistive technologies and devices.
- Production SaaS needs authenticated individual users, tenancy, traffic controls, observability, regulatory/privacy work, and clinician sign-off.

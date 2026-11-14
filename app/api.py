"""FastAPI API and same-origin premium web frontend.

For public deployment: configure a strong ADMIN_API_KEY, HTTPS, request limits,
legal permission for guideline content, and independent clinical validation.
"""
from __future__ import annotations
import hmac
import json
import os
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .catalog import CATALOG
from .config import Settings
from .openrouter import OpenRouterError
from .rag import MedicalRAG
from .evaluation import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'ui' / 'web'
app = FastAPI(
    title='Clinical Guideline Intelligence', version='2.0.0',
    description='Evidence-first clinical guideline question answering; not individualized medical advice.',
)
if (WEB / 'assets').is_dir():
    app.mount('/assets', StaticFiles(directory=str(WEB / 'assets')), name='assets')

@lru_cache
def get_rag():
    return MedicalRAG(Settings())


def protect_write(x_admin_key: str | None = Header(default=None)):
    required = get_rag().settings.admin_api_key
    if required.startswith('replace_'):
        raise HTTPException(503, 'Set a unique ADMIN_API_KEY in .env before administrative actions')
    if not required and os.getenv('APP_ENV', '').lower() == 'production':
        raise HTTPException(503, 'Admin access is not configured. Set ADMIN_API_KEY before deployment.')
    if required and not hmac.compare_digest(x_admin_key or '', required):
        raise HTTPException(403, 'Admin key is missing or incorrect')


class Question(BaseModel):
    question: str = Field(min_length=0, max_length=1500)

class EvalRequest(BaseModel):
    cases: list[dict] = Field(min_length=1, max_length=100)
    include_generation: bool = False
    judge: bool = False

class BootstrapRequest(BaseModel):
    guideline_id: str
    rights_confirmed: bool = False

class FeedbackRequest(BaseModel):
    vote: str = Field(pattern='^(helpful|not_helpful)$')


def handle_error(exc: Exception):
    if isinstance(exc, ValueError):
        raise HTTPException(400, str(exc)) from exc
    if isinstance(exc, OpenRouterError):
        raise HTTPException(503, str(exc)) from exc
    raise exc


@app.get('/', include_in_schema=False)
def index():
    if not (WEB / 'index.html').exists():
        raise HTTPException(503, 'Web frontend missing; open /docs for API.')
    return FileResponse(WEB / 'index.html')


@app.get('/favicon.svg', include_in_schema=False)
def favicon():
    return FileResponse(WEB / 'favicon.svg', media_type='image/svg+xml')


@app.get('/health')
def health():
    rag = get_rag()
    docs = rag.store.list_documents()
    return {'status': 'ok', 'documents': len(docs), 'chunks': sum(d['chunk_count'] for d in docs),
            'provider': 'OpenRouter', 'configured': bool(rag.settings.api_key and rag.settings.api_key != 'replace_with_your_key'),
            'ready': bool(docs and rag.settings.api_key and rag.settings.api_key != 'replace_with_your_key'),
            'medical_use': 'educational_reference_only', 'version': '2.0.0'}


@app.get('/api/system')
def system_info():
    rag = get_rag()
    return {'chat_model': rag.settings.chat_model, 'embedding_model': rag.settings.embed_model,
            'top_k': rag.settings.top_k, 'min_similarity': rag.settings.min_similarity,
            'admin_configured': bool(rag.settings.admin_api_key and not rag.settings.admin_api_key.startswith('replace_')), 'production_mode': os.getenv('APP_ENV','').lower() == 'production'}


@app.get('/documents')
def documents():
    return get_rag().store.list_documents()


@app.get('/api/catalog')
def catalog():
    docs = get_rag().store.list_documents()
    urls = {x['source_url'] for x in docs}
    return [{**g.public(), 'indexed': g.pdf_url in urls} for g in CATALOG.values()]


@app.post('/documents/upload', dependencies=[Depends(protect_write)])
async def upload(file: UploadFile = File(...), title: str = Form(''), source_url: str = Form(''), rights_confirmed: bool = Form(False)):
    rag = get_rag()
    if os.getenv('APP_ENV', '').lower() == 'production' and not rights_confirmed:
        raise HTTPException(400, 'You must confirm lawful reuse rights and that the PDF contains no identifiable patient data')
    if source_url:
        parsed = urlsplit(source_url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
            raise HTTPException(400, 'Source URL must be a valid HTTPS URL')
    contents = await file.read(rag.settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        return rag.ingest(contents, file.filename or 'uploaded.pdf', title=title, source_url=source_url)
    except Exception as exc:
        handle_error(exc)


def download_pdf(url: str, max_bytes: int) -> bytes:
    # Called ONLY with publisher URLs from the fixed allowlist, never a user URL.
    try:
        with httpx.Client(timeout=httpx.Timeout(110, connect=15), follow_redirects=True) as client:
            with client.stream('GET', url, headers={'User-Agent': 'ClinicalGuidelineIntelligence/2.0'}) as res:
                res.raise_for_status()
                payload = bytearray()
                for part in res.iter_bytes(128 * 1024):
                    payload.extend(part)
                    if len(payload) > max_bytes:
                        raise ValueError('Publisher PDF exceeds upload size limit')
                if not payload.startswith(b'%PDF'):
                    raise ValueError('Publisher did not return a valid PDF; use manual upload')
                return bytes(payload)
    except httpx.HTTPError as exc:
        raise ValueError(f'Publisher download unavailable ({type(exc).__name__}). Try again later or upload an authorised copy.') from exc


@app.post('/api/catalog/import', dependencies=[Depends(protect_write)])
def import_guideline(request: BootstrapRequest):
    guid = CATALOG.get(request.guideline_id)
    if guid is None:
        raise HTTPException(404, 'Not a supported publisher guideline')
    if not request.rights_confirmed:
        raise HTTPException(400, 'Review the source licence and confirm you have the necessary rights before import')
    rag = get_rag()
    if not rag.settings.api_key or rag.settings.api_key == 'replace_with_your_key':
        raise HTTPException(503, 'OpenRouter API key not configured in backend .env')
    try:
        content = download_pdf(guid.pdf_url, rag.settings.max_upload_mb * 1024 * 1024)
        return rag.ingest(content, f'{guid.id}.pdf', title=f'{guid.publisher}  {guid.title} ({guid.year})', source_url=guid.pdf_url)
    except Exception as exc:
        handle_error(exc)


@app.get('/documents/{document_id}/pdf')
def source_pdf(document_id: str):
    # Serve only indexed document IDs, and never accept a client-supplied path.
    indexed = {d['id'] for d in get_rag().store.list_documents()}
    if document_id not in indexed:
        raise HTTPException(404, 'Indexed PDF not found')
    path = Path(get_rag().settings.pdf_dir) / f'{document_id}.pdf'
    if not path.is_file():
        raise HTTPException(404, 'Original PDF not stored for this document')
    return FileResponse(path, media_type='application/pdf', headers={'X-Content-Type-Options': 'nosniff'})


@app.delete('/documents/{document_id}', dependencies=[Depends(protect_write)])
def remove_document(document_id: str):
    rag = get_rag()
    if not rag.store.delete_document(document_id):
        raise HTTPException(404, 'Document not found')
    (Path(rag.settings.pdf_dir) / f'{document_id}.pdf').unlink(missing_ok=True)
    return {'deleted': True, 'id': document_id}


@app.post('/retrieve')
def retrieve(request: Question):
    try:
        return {'results': get_rag().retrieve(request.question)}
    except Exception as exc:
        handle_error(exc)


@app.post('/chat')
def chat(request: Question):
    try:
        return get_rag().chat(request.question)
    except Exception as exc:
        handle_error(exc)


@app.post('/api/feedback')
def feedback(request: FeedbackRequest):
    get_rag().store.record_feedback(request.vote)
    return {'saved': True}


@app.get('/api/feedback/stats', dependencies=[Depends(protect_write)])
def feedback_stats():
    return get_rag().store.feedback_stats()


@app.get('/api/evaluation/cases')
def evaluation_cases():
    return json.loads((ROOT / 'data' / 'eval_nice.json').read_text(encoding='utf-8'))


@app.post('/evaluation/run', dependencies=[Depends(protect_write)])
def evaluation(request: EvalRequest):
    try:
        return run_evaluation(get_rag(), request.cases, include_generation=request.include_generation, judge=request.judge)
    except Exception as exc:
        handle_error(exc)

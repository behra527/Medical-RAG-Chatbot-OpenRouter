"""Friendly frontend. Keep backend local unless authentication/TLS deployed."""
from __future__ import annotations
import json
import os
from pathlib import Path
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv('API_URL', 'http://127.0.0.1:8000').rstrip('/')

st.set_page_config(
    page_title='Veritas Health | Clinical RAG AI',
    page_icon='⚕️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Clean minimalist CSS without text artifacts
st.markdown('''<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1140px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Status Pill */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-badge.offline {
    background: rgba(239, 68, 68, 0.12);
    color: #dc2626;
    border-color: rgba(239, 68, 68, 0.3);
}

/* Citation Cards */
.citation-card {
    border-left: 4px solid #0284c7;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.meta-pill {
    background: rgba(2, 132, 199, 0.15);
    color: #0284c7;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
}
</style>''', unsafe_allow_html=True)


def request(method: str, route: str, **kwargs):
    headers = kwargs.pop('headers', {})
    key = st.session_state.get('admin_key', '')
    if key:
        headers['X-Admin-Key'] = key
    try:
        with httpx.Client(timeout=240) as client:
            r = client.request(method, f'{API}{route}', headers=headers, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error('Backend unreachable. Ensure FastAPI is running at http://127.0.0.1:8000')
    except httpx.HTTPStatusError as exc:
        try:
            message = exc.response.json().get('detail', 'Request failed')
        except ValueError:
            message = f'HTTP {exc.response.status_code}'
        st.error(str(message))
    except httpx.HTTPError as exc:
        st.error(f'Request error: {type(exc).__name__}')
    return None


# --- SIDEBAR ---
with st.sidebar:
    st.markdown('### ⚕️ Veritas Health')
    st.caption('Clinical Evidence Workspace')
    
    page = st.radio(
        'Navigation',
        ['Clinical Inquiry', 'Guideline Repository', 'Evaluation Suite', 'System Architecture'],
        label_visibility='collapsed'
    )
    
    st.divider()
    
    status = request('GET', '/health')
    if status:
        st.markdown(
            f'<div class="status-badge">● System Online · {status["documents"]} Guideline(s)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="status-badge offline">● Backend Offline</div>', unsafe_allow_html=True)
        
    st.write('')
    with st.expander('Admin Authentication', expanded=False):
        st.text_input('API Key', key='admin_key', type='password', placeholder='Optional X-Admin-Key')

    st.divider()
    st.caption(
        '**Research & Educational Use Only.** Does not constitute clinical advice. '
        'Do not input protected health information (PHI) or patient identifiers.'
    )


# --- HEADER USING NATIVE STREAMLIT (Zero Blur / Zero Fringing) ---
col_logo, col_text = st.columns([0.08, 0.92])
with col_logo:
    st.markdown('### ⚕️')
with col_text:
    st.markdown('## Veritas Health')
    st.markdown('*Clinical evidence workspace • Page-accurate citation retrieval*')

st.divider()


# --- 1. CLINICAL INQUIRY (CHAT) ---
if page == 'Clinical Inquiry':
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    selected_prompt = None

    if not st.session_state.chat_history:
        st.markdown('#### Query Indexed Clinical Guidelines')
        st.caption('Select a sample clinical scenario below or enter your own question in English or Urdu.')
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                'How does NICE recommend confirming hypertension if clinic BP is 140/90 to 180/120 mmHg?',
                use_container_width=True
            ):
                selected_prompt = 'How does NICE recommend confirming hypertension if clinic BP is between 140/90 and 180/120 mmHg?'
        with c2:
            if st.button(
                'What are the first-line antihypertensive drug steps for adults under 55 without type 2 diabetes?',
                use_container_width=True
            ):
                selected_prompt = 'What are the first-line antihypertensive drug steps for adults under 55 without type 2 diabetes?'

    for item in st.session_state.chat_history:
        with st.chat_message('user'):
            st.markdown(item['question'])
            
        with st.chat_message('assistant'):
            st.markdown(item['answer'])
            
            if item['sources']:
                with st.expander(f'Source Evidence ({len(item["sources"])} cited passages)', expanded=True):
                    for source in item['sources']:
                        rec_badge = f'<span class="meta-pill">Rec {source["recommendation"]}</span>' if source.get('recommendation') else ''
                        st.markdown(f'''
                        <div class="citation-card" style="background: rgba(100,116,139,0.08);">
                            <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">
                                <span>[{source["id"]}] {source["document"]}</span>
                                <span class="meta-pill">Page {source["page"]}</span>
                                <span class="meta-pill">Sec {source["section"]}</span>
                                {rec_badge}
                                <span style="margin-left:auto; opacity:0.7;">Match: {source["similarity"]:.2f}</span>
                            </div>
                            <div style="font-size: 0.88rem; line-height: 1.5;">{source["excerpt"]}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        if source.get('source_url'):
                            st.caption(f'Official reference: [{source["source_url"]}]({source["source_url"]})')

            status_label = 'Abstained (Insufficient Evidence)' if item['abstained'] else 'Verified Grounding'
            st.caption(f'{status_label} · Latency: {item["response_time_ms"]:.0f} ms')

    user_input = st.chat_input('Ask a clinical guideline question (English or Urdu)...')
    question = selected_prompt or user_input

    if question:
        with st.spinner('Retrieving guideline passages and verifying citations...'):
            data = request('POST', '/chat', json={'question': question})
        if data:
            st.session_state.chat_history.append({'question': question, **data})
            st.rerun()

    if st.session_state.chat_history:
        col_spacer, col_btn = st.columns([5, 1])
        with col_btn:
            if st.button('Clear Session', use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# --- 2. GUIDELINE REPOSITORY ---
elif page == 'Guideline Repository':
    col_upload, col_list = st.columns([1, 1.4], gap='large')

    with col_upload:
        st.markdown('#### Index New Guideline')
        with st.form('upload', border=True):
            uploaded = st.file_uploader('Clinical Guideline PDF (Text-based)', type=['pdf'])
            title = st.text_input('Document Title', placeholder='e.g., NICE NG136 — Hypertension in Adults')
            url = st.text_input('Official Source URL (Optional)', placeholder='https://www.nice.org.uk/guidance/ng136')
            submitted = st.form_submit_button('Process & Index PDF', type='primary', use_container_width=True)
            
        if submitted:
            if not uploaded:
                st.warning('Please select a PDF file before indexing.')
            else:
                with st.spinner('Extracting pages, detecting sections, and generating embeddings...'):
                    res = request(
                        'POST',
                        '/documents/upload',
                        files={'file': (uploaded.name, uploaded.getvalue(), 'application/pdf')},
                        data={'title': title, 'source_url': url}
                    )
                if res:
                    st.success(f'Indexed {res["chunk_count"]} chunks across {res["pages"]} pages.')
                    st.rerun()

    with col_list:
        st.markdown('#### Indexed Knowledge Base')
        docs = request('GET', '/documents')
        if docs is not None:
            if not docs:
                st.info('No clinical guidelines indexed yet. Upload a PDF on the left to begin.')
            for doc in docs:
                with st.container(border=True):
                    c_info, c_del = st.columns([4, 1])
                    with c_info:
                        st.markdown(f'**{doc["title"]}**')
                        st.caption(f'{doc["pages"]} pages · {doc["chunk_count"]} semantic chunks · Indexed {doc["indexed_at"]}')
                        if doc.get('source_url'):
                            st.caption(f'[View Official Publication]({doc["source_url"]})')
                    with c_del:
                        if st.button('Remove', key=doc['id'], use_container_width=True):
                            deleted = request('DELETE', f'/documents/{doc["id"]}')
                            if deleted:
                                st.rerun()


# --- 3. EVALUATION SUITE ---
elif page == 'Evaluation Suite':
    st.markdown('#### Retrieval & Grounding Benchmark')
    st.caption('Evaluate section recall, MRR@k, and citation accuracy against curated clinical test cases.')

    with st.container(border=True):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            dataset_file = st.file_uploader('Custom Evaluation Dataset (JSON)', type=['json'])
            try:
                if dataset_file:
                    cases = json.loads(dataset_file.read())
                else:
                    cases = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'eval_nice.json').read_text('utf-8'))
                st.caption(f'Loaded **{len(cases)}** evaluation test cases.')
            except Exception as exc:
                st.error(f'Invalid dataset format: {exc}')
                cases = []

        with c2:
            st.markdown('**Execution Options**')
            generate = st.checkbox('Evaluate full LLM generation (incurs API cost)', value=False)
            judge = st.checkbox('Enable LLM-as-a-Judge scoring (experimental)', value=False, disabled=not generate)
            run_btn = st.button('Run Benchmark', type='primary', disabled=not cases, use_container_width=True)

    if run_btn:
        with st.spinner('Running retrieval and grounding evaluation...'):
            report = request('POST', '/evaluation/run', json={'cases': cases, 'include_generation': generate, 'judge': judge})
        if report:
            st.session_state['latest_report'] = report

    if 'latest_report' in st.session_state:
        report = st.session_state['latest_report']
        stats = report['metrics']
        
        st.write('')
        a, b, c, d = st.columns(4)
        a.metric('Hit@k Recall', f'{100 * stats["retrieval_hit_at_k"]:.1f}%' if stats['retrieval_hit_at_k'] is not None else 'N/A')
        b.metric('MRR@k', f'{stats["mrr_at_k"]:.3f}' if isinstance(stats['mrr_at_k'], float) else str(stats['mrr_at_k']))
        c.metric('Mean Retrieval', f'{stats["mean_retrieval_ms"]:.0f} ms')
        d.metric('Mean Generation', f'{stats["mean_answer_ms"]:.0f} ms' if stats['mean_answer_ms'] is not None else 'Skipped')

        st.markdown('#### Case-Level Breakdown')
        st.dataframe(
            [{k: v for k, v in row.items() if k not in ('sources', 'answer')} for row in report['cases']],
            use_container_width=True,
            hide_index=True
        )
        
        col_dl, col_warn = st.columns([1, 2.5])
        with col_dl:
            st.download_button(
                'Export Full Report (JSON)',
                data=json.dumps(report, indent=2),
                file_name='clinical_rag_evaluation.json',
                mime='application/json',
                use_container_width=True
            )
        with col_warn:
            st.caption('Note: Automated lexical/retrieval metrics do not replace formal clinician adjudication.')


# --- 4. SYSTEM ARCHITECTURE ---
else:
    st.markdown('#### Deterministic Retrieval & Citation Pipeline')
    st.caption('Strict metadata preservation prevents hallucinated page numbers or ungrounded clinical recommendations.')

    steps = [
        ('1. Document Ingestion & Structural Parsing', 'Page-level PDF extraction with hierarchical clinical section and recommendation numbering detection.'),
        ('2. Semantic Chunking & Metadata Binding', 'Overlapping text segments permanently bound to source document ID, exact PDF page, and section tags.'),
        ('3. Hybrid Indexing (SQLite Vector + FTS5)', 'Dense embeddings via OpenRouter combined with lexical BM25 full-text search index.'),
        ('4. Reciprocal Rank Fusion (RRF) & Similarity Gate', 'Fuses dense cosine similarity and BM25 rankings; enforces MIN_SIMILARITY threshold to trigger abstention on out-of-scope queries.'),
        ('5. Constrained Generation & Citation Validation', 'LLM synthesizes answers strictly from retrieved context and validates [S#] citation IDs against source metadata.')
    ]

    for title_step, desc in steps:
        with st.container(border=True):
            st.markdown(f'**{title_step}**')
            st.caption(desc)
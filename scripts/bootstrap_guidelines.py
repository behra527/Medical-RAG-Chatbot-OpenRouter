"""Explicit publisher PDF bootstrap from a developer shell (paid embedding calls)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.catalog import CATALOG
from app.api import download_pdf
from app.rag import MedicalRAG
from app.config import Settings


def main():
    p = argparse.ArgumentParser(description='Download and index a verified official clinical guideline.')
    p.add_argument('--guideline', choices=CATALOG, default='who-hypertension-2021', help='Official publisher source ID')
    p.add_argument('--confirm-rights', action='store_true', help='Confirm you reviewed and have the necessary usage rights')
    args = p.parse_args()
    g = CATALOG[args.guideline]
    print(f'Publisher: {g.publisher}\nTitle: {g.title}\nRights: {g.license_note}\nOfficial source: {g.info_url}')
    if not args.confirm_rights:
        p.error('Review rights above, then rerun with --confirm-rights. No download was attempted.')
    settings = Settings()
    if not settings.api_key or settings.api_key == 'replace_with_your_key':
        p.error('Set a valid OPENROUTER_API_KEY in .env before indexing (embeddings are billed).')
    print('Downloading official PDF and producing OpenRouter embeddings…')
    contents = download_pdf(g.pdf_url, settings.max_upload_mb * 1024 * 1024)
    rag = MedicalRAG(settings)
    result = rag.ingest(contents, f'{g.id}.pdf', title=f'{g.publisher} — {g.title} ({g.year})', source_url=g.pdf_url)
    print(f'{result["status"]}: {result["pages"]} PDF pages, {result["chunk_count"]} searchable passages.')
    print('The original publisher PDF is preserved in the local data/guidelines folder for verified page citations.')


if __name__ == '__main__':
    main()

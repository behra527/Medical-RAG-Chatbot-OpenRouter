"""Download official NICE NG136 PDF; then upload via UI or --index it."""
from pathlib import Path
import argparse
import sys
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag import MedicalRAG
from app.config import Settings

NICE_URL = ('https://www.nice.org.uk/guidance/ng136/resources/'
            'hypertension-in-adults-diagnosis-and-managementpdf-66141722710213')
TARGET = ROOT / 'data' / 'guidelines' / 'nice_ng136_hypertension_2026.pdf'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true', help='Also generate paid OpenRouter embeddings and index')
    parser.add_argument('--confirm-rights', action='store_true', help='Confirm permission for your intended use of NICE content')
    args = parser.parse_args()
    if not args.confirm_rights:
        parser.error('Review NICE terms first, then rerun with --confirm-rights')
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        response = client.get(NICE_URL, headers={'User-Agent': 'ClinicalRAG/1.0 (educational)'})
        response.raise_for_status()
        data = response.content
    if not data.startswith(b'%PDF'):
        raise RuntimeError('Source did not return a PDF. Download manually from NICE and use upload UI.')
    TARGET.write_bytes(data)
    print(f'Official NICE PDF downloaded to {TARGET} ({len(data)} bytes)')
    if args.index:
        result = MedicalRAG(Settings()).ingest(data, TARGET.name,
            title='NICE NG136 — Hypertension in Adults (2026)', source_url=NICE_URL)
        print(f'Indexed {result["chunk_count"]} chunks, status={result["status"]}')


if __name__ == '__main__':
    main()

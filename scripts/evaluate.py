from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag import MedicalRAG
from app.config import Settings
from app.evaluation import load_dataset, run_evaluation


def main():
    parser = argparse.ArgumentParser(description='Run measured clinical RAG evaluation')
    parser.add_argument('--dataset', default=str(ROOT / 'data' / 'eval_nice.json'))
    parser.add_argument('--retrieval-only', action='store_true', help='Skip the LLM and answer metrics')
    parser.add_argument('--judge', action='store_true', help='Use LLM-as-judge; extra API cost')
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    report = run_evaluation(MedicalRAG(Settings()), cases, include_generation=not args.retrieval_only, judge=args.judge and not args.retrieval_only)
    out = ROOT / 'data' / 'reports' / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))
    print(f'Full per-question report: {out}')


if __name__ == '__main__':
    main()

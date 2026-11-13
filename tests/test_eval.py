from app.evaluation import run_evaluation


def test_retrieval_evaluation_and_answer_metrics(rag):
    cases = [
        {'question': 'How to diagnose blood pressure in adults?', 'gold_document': 'NICE NG136',
         'gold_section': '1.2', 'expected_terms': ['ABPM']},
        {'question': 'How do I fix my car engine?', 'should_abstain': True}
    ]
    report = run_evaluation(rag, cases)
    m = report['metrics']
    assert m['retrieval_hit_at_k'] == 1.0
    assert m['mrr_at_k'] == 1.0
    assert m['abstention_accuracy'] == 1.0
    assert m['expected_term_coverage'] == 1.0
    assert m['citation_presence_rate'] == 1.0
    assert len(report['cases']) == 2


def test_retrieval_only(rag):
    report = run_evaluation(rag, [{'question':'What blood pressure measurement?'}], include_generation=False)
    assert report['metrics']['mean_answer_ms'] is None

"""Keep critic costs separate and do not equate missing gold with wrong answers."""
import numpy as np
import pytest
from scripts.analyze_corpus_reader import summarize, paired


def row(identifier, em, complete=False, recall=0.5):
    return {'id': identifier, 'em': em, 'f1': em, 'latency_seconds': 7.2,
        'generated_tokens': 30, 'retrieval_seconds': 0.2, 'num_hops': 1,
        'stop_reason': 'fixed_evidence_budget', 'predicted_sufficient': True,
        'retrieval_metrics': {'complete_support': complete, 'sentence_recall': recall},
        'calls': [dict(role='reader', latency_seconds=2., generated_tokens=10, parse_failed=False, strict_json=True),
                  dict(role='sufficiency_critic', latency_seconds=5., generated_tokens=20, parse_failed=False, strict_json=False)]}


def test_proxy_is_not_correctness_and_roles_are_separate():
    m = summarize([row('a', 1), row('b', 0, True)])
    assert m['em'] == .5
    assert m['by_annotation_coverage']['False']['em'] == 1
    assert m['mean_reader_plus_retrieval_seconds'] == 2.2
    assert m['roles']['sufficiency_critic']['mean_generated_tokens'] == 20
    assert m['roles']['sufficiency_critic']['non_strict_json_rate'] == 1
    assert m['critic_vs_annotation_proxy']['counts']['critic_True_coverage_False'] == 1


def test_pairing_retains_harms_and_rejects_reordering():
    base = [row('a', 0), row('b', 1)]
    improved = [row('a', 1, recall=1), row('b', 0, recall=1)]
    p = paired(improved, base, np.array([[0, 1], [1, 0]]))
    assert p['em']['mean_delta'] == 0
    assert p['em']['improved_ids'] == ['a']
    assert p['em']['regressed_ids'] == ['b']
    assert p['evidence_gain_without_em_gain'] == ['b']
    with pytest.raises(ValueError):
        paired(improved[::-1], base, np.array([[0, 1]]))

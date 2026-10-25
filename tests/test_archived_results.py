from tools.verify_paper_results import verify

def test_archived_summary_matches_underlying_distributions():
    rows=verify()
    assert any('Literature Fusion' in r['method'] for r in rows)
    assert {r['cohort'] for r in rows} == {'all_1063','long_759'}

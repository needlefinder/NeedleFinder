import numpy as np
import pytest
from NeedleFinderLib.EvaluationMetrics import compute_unified_pair_metrics, evaluate_method_unified


def line(x=0, y=0, start=0, stop=30):
    return np.array([[x, y, start], [x, y, stop]], dtype=float)


def test_physical_tip_and_shaft_errors():
    m = compute_unified_pair_metrics(line(3, 4, stop=42), line())
    assert m['mean_shaft_xy'] == 5
    assert m['tip_xy'] == 5
    assert m['tip_z'] == 12
    assert m['tip_3d'] == 13
    assert compute_unified_pair_metrics(line()[::-1], line())['tip_3d'] == 0


@pytest.mark.parametrize('pred,gt', [(line(stop=1), line()), (line(start=40, stop=60), line()), (line(9), line())])
def test_unmatchable_pairs(pred, gt):
    assert compute_unified_pair_metrics(pred, gt) is None


@pytest.mark.parametrize('preds,gts,matched,precision,recall', [
    ([], [], 0, 0, 0), ([], [line()], 0, 0, 0),
    ([line()], [], 0, 0, 0), ([line(), line()], [line()], 1, 50, 100),
    ([line()], [line(), line()], 1, 100, 50),
    ([line(3.5)], [line()], 1, 100, 100),
    ([line(3.5001)], [line()], 0, 0, 0),
])
def test_matching_empty_duplicates_and_threshold(preds, gts, matched, precision, recall):
    m, _ = evaluate_method_unified([preds], [gts], n_boot=10)
    assert m['hungarian_1to1_matched'] == matched
    assert m['precision_1to1_pct'] == precision
    assert m['hungarian_1to1_recall_pct'] == recall


def test_assignment_maximizes_valid_matches_before_distance():
    # Costs [[0, 3], [3, 4.24]]: unconstrained Hungarian chooses one invalid pair.
    m, _ = evaluate_method_unified([[line(), line(0, 3)]], [[line(), line(3)]], n_boot=10)
    assert m['hungarian_1to1_matched'] == 2


def test_auxiliary_obturator_matching_cannot_excuse_duplicate_predictions():
    m, _ = evaluate_method_unified([[line(), line(6), line(6)]], [[line()]], [[line(6)]], n_boot=10)
    assert m['aux_short_obturator_matched'] == 1
    assert m['precision_1to1_obturator_adj_pct'] == 50


def test_patient_bootstrap_is_repeatable():
    args = ([[line()], []], [[line()], [line()]])
    assert evaluate_method_unified(*args, n_boot=20) == evaluate_method_unified(*args, n_boot=20)


def test_single_point_pair_has_no_valid_shaft():
    assert compute_unified_pair_metrics(np.array([[0,0,0]]),line()) is None


def test_no_bootstrap_requested_has_empty_intervals():
    result,_=evaluate_method_unified([[line()]],[[line()]],n_boot=0)
    assert result['bootstrap_95ci_1000_patients']['f1_1to1_pct']==[0,0]

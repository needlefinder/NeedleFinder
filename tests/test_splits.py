import pytest
from NeedleFinderLib.DatasetSplits import nested_split, validate_checkpoint_provenance


def test_all_25_nested_folds_partition_patients_without_leakage():
    ids = [f'patient-{i}' for i in range(54)]
    outer_seen = []
    for outer in range(5):
        inner_seen = []
        for inner in range(5):
            s = nested_split(ids, outer, inner)
            train, val, test = (set(s[k]) for k in ('inner_train', 'inner_val', 'outer_test'))
            assert not (train & val or train & test or val & test)
            assert train | val | test == set(range(54))
            assert s['outer_test'] == [i for i in range(54) if i % 5 == outer]
            assert s['inner_val'] == s['outer_train'][inner::5]
            inner_seen.extend(s['inner_val'])
        assert sorted(inner_seen) == s['outer_train']
        outer_seen.extend(s['outer_test'])
    assert sorted(outer_seen) == list(range(54))


@pytest.mark.parametrize('ids,outer,inner', [(['a']*54,0,0), (list(range(54)),5,0), (list(range(54)),0,-1), ([1,2],0,0)])
def test_invalid_splits_fail(ids, outer, inner):
    with pytest.raises(ValueError): nested_split(ids, outer, inner)


@pytest.mark.parametrize('key', ['training_indices', 'selection_indices', 'calibration_indices'])
def test_test_patient_cannot_train_select_or_calibrate(key):
    s = nested_split(list(range(54)), 0, 0)
    record = dict(training_indices=s['inner_train'], selection_indices=s['inner_val'],
                  calibration_indices=s['inner_val'], checkpoint_sha256='a'*64)
    validate_checkpoint_provenance(record, s)
    record[key] = [s['outer_test'][0]]
    with pytest.raises(ValueError): validate_checkpoint_provenance(record, s)


def test_frozen_patient_order_and_splits_match_training_protocol():
    import json
    from pathlib import Path
    spec=json.loads((Path(__file__).parents[1]/'reproducibility/splits-54.json').read_text())
    assert len(spec['patient_ids'])==54
    for outer in range(5):
        for inner in range(5):
            actual=nested_split(spec['patient_ids'],outer,inner)
            expected=spec['folds'][f'outer{outer}_inner{inner}']
            assert all(actual[k]==expected[k] for k in actual)
            assert expected['seed']==1000+10*outer+inner

@pytest.mark.parametrize('ids,folds,outer,inner', [(list(range(54)),1,0,0), (list(range(5)),5,0,4)])
def test_degenerate_nested_partitions_fail(ids,folds,outer,inner):
    with pytest.raises(ValueError):nested_split(ids,outer,inner,n_folds=folds)

@pytest.mark.parametrize('patch', [{'training_indices':[]},{'checkpoint_sha256':''},{'checkpoint_sha256':'short'}, {'training_indices':[1,1]}])
def test_incomplete_checkpoint_provenance_fails(patch):
    s=nested_split(list(range(54)),0,0)
    record=dict(training_indices=s['inner_train'],selection_indices=s['inner_val'],calibration_indices=s['inner_val'],checkpoint_sha256='a'*64)
    record.update(patch)
    with pytest.raises(ValueError):validate_checkpoint_provenance(record,s)

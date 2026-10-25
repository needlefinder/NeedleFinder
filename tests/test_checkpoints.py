import pytest
from reproducibility.checkpoints import write_checkpoint_record, read_checkpoint_record
from NeedleFinderLib.DatasetSplits import nested_split

def test_cached_weights_require_matching_split_seed_order_and_digest(tmp_path):
    p=tmp_path/'weights.pt';p.write_bytes(b'weights')
    ids=list(range(54));split=nested_split(ids,0,0)
    with pytest.raises(ValueError):read_checkpoint_record(p,split,ids,1000)
    write_checkpoint_record(p,split,ids,1000)
    assert read_checkpoint_record(p,split,ids,1000)['seed']==1000
    with pytest.raises(ValueError):read_checkpoint_record(p,split,ids[::-1],1000)
    with pytest.raises(ValueError):read_checkpoint_record(p,split,ids,1001)
    with pytest.raises(ValueError):read_checkpoint_record(p,nested_split(ids,0,1),ids,1000)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):read_checkpoint_record(p,split,ids,1000)

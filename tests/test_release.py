import json
import pytest
from tools import release


def test_artifact_tampering_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    model=tmp_path/'model.bin';model.write_bytes(b'original')
    lock={'files':{'model.bin':{'bytes':8,'sha256':release.sha256(model)}}}
    assert release.verify_artifacts(lock)==1
    model.write_bytes(b'modified')
    with pytest.raises(ValueError): release.verify_artifacts(lock)


def test_empty_or_escaping_artifact_lock_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    for lock in [{'files':{}},{'files':{'../model.bin':{}}}]:
        with pytest.raises(ValueError): release.verify_artifacts(lock)


def test_release_requires_all_fresh_non_skipped_reports(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'git',lambda *args:'')
    monkeypatch.setattr(release,'source_fingerprint',lambda:'current')
    assert len(release.release_blockers(tmp_path))==len(release.REQUIRED_CHECKS)
    for name in release.REQUIRED_CHECKS:
        report=tmp_path/f'{name}.xml';report.write_text('<report/>')
        record=dict(status='passed',source_fingerprint='current',skipped=0,
                    report=report.name,report_sha256=release.sha256(report))
        (tmp_path/f'{name}.json').write_text(json.dumps(record))
    assert release.release_blockers(tmp_path)==[]
    record['source_fingerprint']='stale';record['skipped']=1
    (tmp_path/'fast.json').write_text(json.dumps(record))
    assert len(release.release_blockers(tmp_path))==2

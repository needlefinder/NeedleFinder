"""Checkpoint provenance for newly trained nested models; never infer old provenance."""
import hashlib
import json
from pathlib import Path


def checkpoint_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_checkpoint_record(path, split, patient_ids, seed):
    record={'training_indices':split['inner_train'],'selection_indices':split['inner_val'],
            'calibration':'not performed by this Stage 1 trainer','patient_ids':list(patient_ids),'seed':seed,
            'checkpoint_sha256':checkpoint_sha256(path)}
    Path(str(path)+'.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def read_checkpoint_record(path, split, patient_ids, seed):
    sidecar=Path(str(path)+'.provenance.json')
    if not sidecar.is_file():
        raise ValueError(f'No provenance for cached checkpoint {path}; use a new output directory to retrain. Do not invent provenance for historical weights.')
    record=json.loads(sidecar.read_text())
    if record.get('patient_ids')!=list(patient_ids) or record.get('seed')!=seed:
        raise ValueError('Checkpoint patient ordering or seed differs from this run')
    for field,part in [('training_indices','inner_train'),('selection_indices','inner_val')]:
        if record.get(field)!=split[part]:raise ValueError(f'Checkpoint split mismatch: {field}')
    if record.get('checkpoint_sha256')!=checkpoint_sha256(path):raise ValueError('Checkpoint checksum mismatch')
    return record

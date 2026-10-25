"""Deterministic nested folds used by the 54-case paper training protocol."""

def nested_split(patient_ids, outer_fold, inner_fold, n_folds=5):
    """Return indices, preserving the published index-modulo outer split.

    Patient IDs must be unique: multiple scans from one patient require a grouped
    protocol rather than silently treating scans as independent patients.
    """
    ids = list(patient_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs: split at patient level before training")
    if n_folds < 2 or len(ids) < n_folds:
        raise ValueError("Need at least two folds and one patient per fold")
    if not (0 <= outer_fold < n_folds and 0 <= inner_fold < n_folds):
        raise ValueError("Fold index out of range")
    test = [i for i in range(len(ids)) if i % n_folds == outer_fold]
    train = [i for i in range(len(ids)) if i % n_folds != outer_fold]
    val = train[inner_fold::n_folds]
    subtrain = [i for i in train if i not in val]
    if not val or not subtrain:
        raise ValueError("Insufficient patients for nested training/validation")
    return {"outer_test": test, "outer_train": train,
            "inner_val": val, "inner_train": subtrain}


def validate_checkpoint_provenance(record, split):
    """Reject missing or contaminated training/selection provenance."""
    for key, allowed in [("training_indices", "inner_train"),
                         ("selection_indices", "inner_val"),
                         ("calibration_indices", "inner_val")]:
        values = record.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Missing {key}")
        if len(values) != len(set(values)) or not set(values) <= set(split[allowed]):
            raise ValueError(f"Invalid/leaked {key}")
    if not record.get("checkpoint_sha256") or len(record["checkpoint_sha256"]) != 64:
        raise ValueError("Missing checkpoint SHA-256")

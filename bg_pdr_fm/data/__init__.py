"""Data adapters and batch helpers for BG-PDR-FM.

The package exports the same public names as before, but resolves torch-heavy
objects lazily. This keeps lightweight utilities such as LMDB packing usable in
environments where only NumPy/LMDB dependencies are installed.
"""

from __future__ import annotations

_EXPORTS = {
    "batch": {
        "BGInferenceBatch",
        "BGSampleBatch",
        "INFERENCE_FORBIDDEN_KEYS",
        "REQUIRED_INFERENCE_KEYS",
        "STANDARD_MODALITIES",
        "batch_to_device",
        "collate_bg_inference_samples",
        "collate_bg_samples",
        "inference_batch_from_mapping",
        "validate_bg_inference_batch",
        "validate_bg_batch",
    },
    "datasets": {
        "DEFAULT_OPENFWI_DATASETS",
        "MarmousiBGDataset",
        "OpenFWIBGDataset",
        "SyntheticBGDataset",
        "resolve_marmousi_root",
        "resolve_openfwi_root",
    },
    "raw_datasets": {
        "Marmousi",
        "OpenFWI",
    },
}

__all__ = sorted(name for names in _EXPORTS.values() for name in names)


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = __import__(f"{__name__}.{module_name}", fromlist=[name])
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

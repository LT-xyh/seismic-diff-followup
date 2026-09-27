"""External GFI baseline bridge.

This package intentionally does not reimplement GFI. It records the official
repository contract and provides small utilities for smoke/data-interface
checks, post-hoc metric evaluation, and tensor-contract wrappers.
"""

__all__ = [
    "GFIAnnotationDataset",
    "GFIModelWrapper",
    "AdaptedGFIMultimodal",
    "AdaptedGFILatentUNetMultimodal",
    "annotation_summary",
    "build_adapted_gfi_input",
    "build_gfi_dataloader",
    "evaluate_arrays",
    "read_gfi_annotation",
]


def __getattr__(name: str) -> object:
    if name in {"GFIAnnotationDataset", "annotation_summary", "build_gfi_dataloader", "read_gfi_annotation"}:
        from . import adapter

        return getattr(adapter, name)
    if name == "evaluate_arrays":
        from .evaluate_gfi_predictions import evaluate_arrays

        return evaluate_arrays
    if name == "GFIModelWrapper":
        from .model_wrapper import GFIModelWrapper

        return GFIModelWrapper
    if name in {"AdaptedGFIMultimodal", "build_adapted_gfi_input"}:
        from . import adapted_multimodal

        return getattr(adapted_multimodal, name)
    if name == "AdaptedGFILatentUNetMultimodal":
        from .adapted_latent_unet import AdaptedGFILatentUNetMultimodal

        return AdaptedGFILatentUNetMultimodal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Adapted Auto-Linear external baseline components."""

__all__ = [
    "AdaptedAutoLinearMultimodal",
    "AdaptedAutoLinearOriginalMultimodal",
    "build_auto_linear_measurement_input",
    "build_auto_linear_original_measurement_input",
]


def __getattr__(name: str) -> object:
    if name in {"AdaptedAutoLinearMultimodal", "build_auto_linear_measurement_input"}:
        from . import adapted_multimodal

        return getattr(adapted_multimodal, name)
    if name in {"AdaptedAutoLinearOriginalMultimodal", "build_auto_linear_original_measurement_input"}:
        from . import original_architecture

        return getattr(original_architecture, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

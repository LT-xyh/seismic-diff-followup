"""Lazy training entrypoint exports for BG-PDR-FM."""


def train_bg_pdr_fm_main(*args, **kwargs):
    from .train_bg_pdr_fm import main

    return main(*args, **kwargs)


def train_autoencoder_main(*args, **kwargs):
    from .train_autoencoder import main

    return main(*args, **kwargs)


def run_contrastive(*args, **kwargs):
    from .run_contrastive import run

    return run(*args, **kwargs)


def run_background(*args, **kwargs):
    from .run_background import run

    return run(*args, **kwargs)


def run_residual(*args, **kwargs):
    from .run_residual import run

    return run(*args, **kwargs)


__all__ = [
    "run_background",
    "run_contrastive",
    "run_residual",
    "train_autoencoder_main",
    "train_bg_pdr_fm_main",
]

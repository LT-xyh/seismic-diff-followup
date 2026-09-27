"""Programmatic launcher for BG-PDR-FM contrastive training.

Run this file directly from Python. All trainable settings come from
`bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml`; edit that YAML to tune.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.training.train_bg_pdr_fm import load_stage_config, run_stage_config


CONFIG_PATH = REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_contrastive.yaml"


def load_contrastive_config():
    return load_stage_config(CONFIG_PATH, "contrastive")


def run() -> None:
    run_stage_config(CONFIG_PATH, "contrastive")


if __name__ == "__main__":
    run()

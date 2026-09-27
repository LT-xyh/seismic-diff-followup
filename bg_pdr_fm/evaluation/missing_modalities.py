"""Official missing-modality protocol for AAAI27 benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from bg_pdr_fm.data.batch import BGSampleBatch, STANDARD_MODALITIES


AAAI27_MISSING_MODES: dict[str, tuple[str, ...]] = {
    "full": STANDARD_MODALITIES,
    "w/o well_log": ("migrated_image", "horizon", "rms_vel"),
    "w/o horizon": ("migrated_image", "rms_vel", "well_log"),
    "w/o rms_vel": ("migrated_image", "horizon", "well_log"),
    "w/o well+rms": ("migrated_image", "horizon"),
    "PSTM only": ("migrated_image",),
}

AAAI27_RMS_ONLY_MISSING_MODES: dict[str, tuple[str, ...]] = {
    "full": STANDARD_MODALITIES,
    "w/o well_log": ("migrated_image", "horizon", "rms_vel"),
    "w/o horizon": ("migrated_image", "rms_vel", "well_log"),
    "w/o rms_vel": ("migrated_image", "horizon", "well_log"),
    "w/o well+rms": ("migrated_image", "horizon"),
    "RMS only": ("rms_vel",),
}

AAAI27_RADAR_MISSING_MODES: dict[str, tuple[str, ...]] = {
    "w/o well_log": ("migrated_image", "horizon", "rms_vel"),
    "w/o horizon": ("migrated_image", "rms_vel", "well_log"),
    "w/o PSTM": ("horizon", "rms_vel", "well_log"),
    "RMS only": ("rms_vel",),
}

AAAI27_SINGLE_MODALITY_MISSING_MODES: dict[str, tuple[str, ...]] = {
    "RMS only": ("rms_vel",),
    "Horizon only": ("horizon",),
    "Well only": ("well_log",),
    "PSTM only": ("migrated_image",),
}


@dataclass(frozen=True)
class MissingModalityProtocol:
    """Reusable missing-modality protocol for benchmark diagnostics."""

    modes: dict[str, tuple[str, ...]]
    modalities: tuple[str, ...] = STANDARD_MODALITIES

    @classmethod
    def aaai27(cls) -> "MissingModalityProtocol":
        return cls(modes=AAAI27_MISSING_MODES)

    @classmethod
    def aaai27_rms_only(cls) -> "MissingModalityProtocol":
        """Paper-facing protocol with a numerical-only RMS condition."""

        return cls(modes=AAAI27_RMS_ONLY_MISSING_MODES)

    @classmethod
    def aaai27_radar(cls) -> "MissingModalityProtocol":
        """Selected scenarios used by the four-panel modality radar figure."""

        return cls(modes=AAAI27_RADAR_MISSING_MODES)

    @classmethod
    def aaai27_single_modality(cls) -> "MissingModalityProtocol":
        """Single-input conditions for modality contribution diagnostics."""

        return cls(modes=AAAI27_SINGLE_MODALITY_MISSING_MODES)

    @property
    def mode_names(self) -> list[str]:
        return list(self.modes)

    @property
    def expected_num_modes(self) -> int:
        return len(self.modes)

    def keep_mask(self, mode: str, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if mode not in self.modes:
            raise ValueError(f"Unknown missing-modality mode {mode!r}. Expected one of {self.mode_names}.")
        keep = torch.zeros(len(self.modalities), device=device, dtype=dtype)
        for modality in self.modes[mode]:
            keep[self.modalities.index(modality)] = 1.0
        return keep

    def apply(self, batch: BGSampleBatch, mode: str) -> BGSampleBatch:
        keep = self.keep_mask(mode, batch.modality_mask.device, batch.modality_mask.dtype)
        keep_b = keep.view(1, -1)
        new_mask = batch.modality_mask * keep_b
        new_quality = batch.modality_quality * keep_b

        inputs = {
            "migrated_image": batch.migrated_image,
            "horizon": batch.horizon,
            "rms_vel": batch.rms_vel,
            "well_log": batch.well_log,
        }
        for idx, modality in enumerate(self.modalities):
            if keep[idx].item() <= 0:
                inputs[modality] = torch.zeros_like(inputs[modality])
        well_mask = batch.well_mask if keep[self.modalities.index("well_log")].item() > 0 else torch.zeros_like(batch.well_mask)

        return replace(
            batch,
            migrated_image=inputs["migrated_image"],
            horizon=inputs["horizon"],
            rms_vel=inputs["rms_vel"],
            well_log=inputs["well_log"],
            well_mask=well_mask,
            modality_mask=new_mask,
            modality_quality=new_quality,
        )


def modality_keep_mask(mode: str, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return MissingModalityProtocol.aaai27().keep_mask(mode, device, dtype)


def apply_missing_modality_mode(batch: BGSampleBatch, mode: str) -> BGSampleBatch:
    return MissingModalityProtocol.aaai27().apply(batch, mode)

"""Normalization profiles shared by BG-PDR-FM datasets and codecs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizationProfile:
    name: str
    ranges: dict[str, tuple[float, float]]
    discrete_modalities: frozenset[str] = frozenset({"horizon", "well_mask"})

    def max_min(self, modality: str) -> tuple[float, float]:
        if modality not in self.ranges:
            raise KeyError(f"Normalization profile {self.name!r} has no range for modality {modality!r}.")
        min_value, max_value = self.ranges[modality]
        return max_value, min_value


NORMALIZATION_PROFILES: dict[str, NormalizationProfile] = {
    "openfwi": NormalizationProfile(
        name="openfwi",
        ranges={
            "depth_vel": (1500.0, 4500.0),
            "time_vel": (1500.0, 4500.0),
            "rms_vel": (1500.0, 4500.0),
            "well_log": (1500.0, 4500.0),
            "migrated_image": (-200.0, 200.0),
            "horizon": (0.0, 1.0),
            "well_mask": (0.0, 1.0),
        },
    ),
    "marmousi": NormalizationProfile(
        name="marmousi",
        ranges={
            "depth_vel": (1000.0, 4700.0),
            "time_vel": (1000.0, 4700.0),
            "well_log": (1000.0, 4700.0),
            "rms_vel": (1500.0, 3100.0),
            "migrated_image": (-0.05, 0.05),
            "horizon": (0.0, 1.0),
            "well_mask": (0.0, 1.0),
        },
    ),
    "seismic_global": NormalizationProfile(
        name="seismic_global",
        ranges={
            "depth_vel": (1000.0, 6000.0),
            "time_vel": (1000.0, 6000.0),
            "rms_vel": (1000.0, 6000.0),
            "well_log": (1000.0, 6000.0),
            "migrated_image": (-200.0, 200.0),
            "horizon": (0.0, 1.0),
            "well_mask": (0.0, 1.0),
        },
    ),
}


def get_normalization_profile(name: str | NormalizationProfile) -> NormalizationProfile:
    if isinstance(name, NormalizationProfile):
        return name
    key = str(name).lower()
    if key not in NORMALIZATION_PROFILES:
        raise ValueError(f"Unknown normalization profile {name!r}. Available: {sorted(NORMALIZATION_PROFILES)}.")
    return NORMALIZATION_PROFILES[key]


def profile_max_min(name: str | NormalizationProfile) -> dict[str, tuple[float, float]]:
    profile = get_normalization_profile(name)
    return {key: profile.max_min(key) for key in profile.ranges}


def normalize_tensor(
    tensor: torch.Tensor,
    modality: str,
    *,
    profile: str | NormalizationProfile,
    mode: str | None = "-1_1",
    clamp: bool = True,
) -> torch.Tensor:
    profile = get_normalization_profile(profile)
    tensor = tensor.to(torch.float32)
    if mode is None or modality in profile.discrete_modalities:
        return tensor

    min_value, max_value = profile.ranges[modality]
    if max_value <= min_value:
        raise ValueError(f"Invalid normalization range for {profile.name}/{modality}: {(min_value, max_value)}.")

    if mode == "01":
        normalized = (tensor - min_value) / (max_value - min_value)
        return normalized.clamp(0.0, 1.0) if clamp else normalized
    if mode == "-1_1":
        normalized = ((tensor - min_value) / (max_value - min_value)) * 2.0 - 1.0
        return normalized.clamp(-1.0, 1.0) if clamp else normalized
    raise ValueError(f"Unsupported normalize mode: {mode!r}.")


def normalize_sparse_well_log(
    well_log: torch.Tensor,
    well_mask: torch.Tensor,
    *,
    profile: str | NormalizationProfile,
    mode: str | None = "-1_1",
    clamp: bool = True,
) -> torch.Tensor:
    if mode is None:
        return well_log.to(torch.float32)

    observed = well_mask > 0
    normalized = torch.zeros_like(well_log, dtype=torch.float32)
    if not observed.any():
        return normalized

    observed_values = normalize_tensor(
        well_log[observed],
        "well_log",
        profile=profile,
        mode=mode,
        clamp=clamp,
    )
    normalized[observed] = observed_values
    return normalized

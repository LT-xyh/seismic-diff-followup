"""Physics-anchored reliability-guided contrastive losses for BG-PDR-FM."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch, STANDARD_MODALITIES
from .types import DecoupledFeatures


MODALITY_PRIORS = {
    "migrated_image": (0.85, 0.45),
    "horizon": (0.95, 0.10),
    "rms_vel": (0.55, 0.90),
    "well_log": (0.25, 0.95),
}


@dataclass
class WaveletAnchorBundle:
    anchor_struct: torch.Tensor
    anchor_num: torch.Tensor


class DepthVelocityWaveletTransform(nn.Module):
    """Fixed Haar wavelet anchors for contrastive calibration."""

    def __init__(
        self,
        level: int = 1,
        boundary: str = "edge",
        resize_mode: str = "nearest",
        detail_fusion: str = "l2",
    ) -> None:
        super().__init__()
        if level < 1:
            raise ValueError(f"Expected level >= 1, got {level}.")
        if resize_mode not in {"nearest", "bilinear"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode}.")
        dec_lo = torch.tensor([1.0 / sqrt(2.0), 1.0 / sqrt(2.0)], dtype=torch.float32)
        dec_hi = torch.tensor([-1.0 / sqrt(2.0), 1.0 / sqrt(2.0)], dtype=torch.float32)
        ll = torch.outer(dec_lo, dec_lo)
        lh = torch.outer(dec_lo, dec_hi)
        hl = torch.outer(dec_hi, dec_lo)
        hh = torch.outer(dec_hi, dec_hi)
        self.level = int(level)
        self.boundary = str(boundary)
        self.resize_mode = str(resize_mode)
        self.detail_fusion = str(detail_fusion)
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1), persistent=False)

    @staticmethod
    def _pad_mode(boundary: str) -> str:
        if boundary in {"edge", "symmetric"}:
            return "replicate"
        if boundary == "reflect":
            return "reflect"
        if boundary in {"wrap", "periodic", "periodization"}:
            return "circular"
        if boundary == "constant":
            return "constant"
        raise ValueError(f"Unsupported boundary mode: {boundary}.")

    def _pad_for_dwt(self, x: torch.Tensor) -> torch.Tensor:
        pad_bottom = x.shape[-2] % 2
        pad_right = x.shape[-1] % 2
        if pad_bottom == 0 and pad_right == 0:
            return x
        mode = self._pad_mode(self.boundary)
        if mode == "constant":
            return F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode, value=0.0)
        return F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode)

    def _single_level(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, _, _ = x.shape
        padded = self._pad_for_dwt(x)
        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        coeffs = F.conv2d(padded, weight, stride=2, padding=0, groups=c)
        coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        if self.resize_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _fuse_details(self, details: Sequence[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(list(details), dim=0)
        if self.detail_fusion == "l2":
            return torch.sqrt(torch.sum(stack.square(), dim=0) + 1e-12)
        if self.detail_fusion == "l1":
            return torch.sum(stack.abs(), dim=0)
        if self.detail_fusion == "mean_abs":
            return torch.mean(stack.abs(), dim=0)
        if self.detail_fusion == "stack_mean":
            return torch.mean(stack, dim=0)
        raise ValueError(f"Unsupported detail_fusion: {self.detail_fusion}.")

    def forward(self, depth_velocity: torch.Tensor) -> WaveletAnchorBundle:
        if depth_velocity.ndim != 4:
            raise ValueError(f"Expected depth velocity as BCHW, got {tuple(depth_velocity.shape)}.")
        original_hw = tuple(depth_velocity.shape[-2:])
        approx = depth_velocity.float()
        detail_maps: list[torch.Tensor] = []
        for _ in range(self.level):
            approx, lh, hl, hh = self._single_level(approx)
            detail_maps.extend(
                [
                    self._resize(lh, original_hw),
                    self._resize(hl, original_hw),
                    self._resize(hh, original_hw),
                ]
            )
        return WaveletAnchorBundle(
            anchor_struct=self._fuse_details(detail_maps),
            anchor_num=self._resize(approx, original_hw),
        )


class VectorProjector(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or embed_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(int(in_channels), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(embed_dim)),
            nn.LayerNorm(int(embed_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if x.ndim != 2:
            raise ValueError(f"Expected BxC vector or BCHW map, got {tuple(x.shape)}.")
        return self.net(x)


def _info_nce_or_cosine(query: torch.Tensor, key: torch.Tensor, temperature: float) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    if query.shape[0] < 2:
        return (1.0 - F.cosine_similarity(query, key, dim=1)).mean()
    logits = query @ key.transpose(0, 1) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


class PhysicsAnchoredSubsetSymileLoss(nn.Module):
    """Reliability-guided Subset-Symile contrastive objective for BG-PDR-FM."""

    def __init__(
        self,
        feature_channels: int = 32,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
        temperature: float = 0.1,
        pairwise_temperature: float = 0.1,
        symile_structural_weight: float = 1.0,
        symile_numerical_weight: float = 1.0,
        pairwise_structural_weight: float = 0.0,
        pairwise_numerical_weight: float = 0.0,
        pairwise_reliability_floor: float = 0.35,
        anchor_weight: float = 0.5,
        reliability_prior_weight: float = 0.05,
        sn_ortho_weight: float = 0.05,
        unique_ortho_weight: float = 0.05,
        structural_highpass_weight: float = 0.0,
        numerical_lowpass_weight: float = 0.0,
        frequency_kernel_size: int = 5,
        min_symile_group_size: int = 2,
        strict_derangement: bool = True,
        num_negative_shuffles: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.pairwise_temperature = float(pairwise_temperature)
        if self.pairwise_temperature <= 0:
            raise ValueError("pairwise_temperature must be positive.")
        self.pairwise_reliability_floor = float(pairwise_reliability_floor)
        if self.pairwise_reliability_floor < 0:
            raise ValueError("pairwise_reliability_floor must be non-negative.")
        self.weights = {
            "symile_structural": float(symile_structural_weight),
            "symile_numerical": float(symile_numerical_weight),
            "pairwise_structural": float(pairwise_structural_weight),
            "pairwise_numerical": float(pairwise_numerical_weight),
            "anchor": float(anchor_weight),
            "reliability_prior": float(reliability_prior_weight),
            "sn_ortho": float(sn_ortho_weight),
            "unique_ortho": float(unique_ortho_weight),
            "structural_highpass": float(structural_highpass_weight),
            "numerical_lowpass": float(numerical_lowpass_weight),
        }
        if frequency_kernel_size % 2 == 0:
            raise ValueError("frequency_kernel_size must be odd.")
        self.frequency_kernel_size = int(frequency_kernel_size)
        self.min_symile_group_size = int(min_symile_group_size)
        self.strict_derangement = bool(strict_derangement)
        self.num_negative_shuffles = (
            None if num_negative_shuffles is None else int(num_negative_shuffles)
        )
        if self.num_negative_shuffles is not None and self.num_negative_shuffles < 1:
            raise ValueError("num_negative_shuffles must be positive when set.")
        item_names = (*STANDARD_MODALITIES, "anchor_struct", "anchor_num")
        self.projectors = nn.ModuleDict(
            {
                name: VectorProjector(
                    feature_channels if name in STANDARD_MODALITIES else 1,
                    embed_dim,
                    hidden_dim,
                )
                for name in item_names
            }
        )
        priors = torch.tensor([MODALITY_PRIORS[name] for name in STANDARD_MODALITIES], dtype=torch.float32)
        self.register_buffer("reliability_prior", priors, persistent=False)

    @staticmethod
    def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    @staticmethod
    def _cos_abs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        av = F.adaptive_avg_pool2d(a, 1).flatten(1)
        bv = F.adaptive_avg_pool2d(b, 1).flatten(1)
        return F.cosine_similarity(av, bv, dim=1, eps=1e-8).abs()

    def _project(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projectors[name](x), dim=1)

    @staticmethod
    def _strict_derangement_indices(batch_size: int, device: torch.device, offset: int = 1) -> torch.Tensor:
        indices = torch.arange(int(batch_size), device=device)
        if batch_size < 2:
            return indices
        shift = int(offset) % int(batch_size)
        if shift == 0:
            shift = 1
        return (indices + shift) % int(batch_size)

    @staticmethod
    def _shuffled_derangement_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        batch_size = int(batch_size)
        indices = torch.arange(batch_size, device=device)
        if batch_size < 2:
            return indices

        shuffled = torch.randperm(batch_size, device=device)
        fixed = shuffled == indices
        if not fixed.any():
            return shuffled

        fixed_positions = torch.nonzero(fixed, as_tuple=False).flatten()
        if fixed_positions.numel() == 1:
            swap_positions = (fixed_positions + 1) % batch_size
            fixed_values = shuffled[fixed_positions].clone()
            shuffled[fixed_positions] = shuffled[swap_positions]
            shuffled[swap_positions] = fixed_values
            return shuffled

        shuffled[fixed_positions] = shuffled[fixed_positions.roll(1)]
        return shuffled

    def _negative_shuffle_count(self, batch_size: int) -> int:
        max_count = max(int(batch_size) - 1, 0)
        if self.num_negative_shuffles is None:
            return max_count
        return min(int(self.num_negative_shuffles), max_count)

    def _mip_shuffle(self, reps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(reps) < 3:
            return self._zero_like_reference(reps[0])
        batch_size = int(reps[0].shape[0])
        if batch_size < 2:
            return self._zero_like_reference(reps[0])
        labels = torch.zeros(batch_size, device=reps[0].device, dtype=torch.long)
        negative_count = self._negative_shuffle_count(batch_size)
        losses = []
        for anchor_idx, anchor in enumerate(reps):
            others = [rep for idx, rep in enumerate(reps) if idx != anchor_idx]
            pos_product = anchor
            for rep in others:
                pos_product = pos_product * rep
            pos_logits = pos_product.sum(dim=1, keepdim=True)

            neg_logits = []
            for _ in range(negative_count):
                neg_product = torch.ones_like(anchor)
                for rep in others:
                    if self.strict_derangement:
                        indices = self._shuffled_derangement_indices(batch_size, rep.device)
                    else:
                        indices = torch.randperm(batch_size, device=rep.device)
                    neg_product = neg_product * rep[indices]
                neg_logits.append((anchor * neg_product).sum(dim=1, keepdim=True))
            logits = torch.cat([pos_logits, *neg_logits], dim=1)
            losses.append(F.cross_entropy(logits / self.temperature, labels))
        return torch.stack(losses).mean()

    def _symile_space(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        reliability: torch.Tensor,
        availability: torch.Tensor,
        anchor: torch.Tensor,
        space: str,
        rel_idx: int,
        anchor_name: str,
    ) -> tuple[torch.Tensor, int, int]:
        reference = anchor
        losses: list[torch.Tensor] = []
        group_count = 0
        sample_count = 0
        for pattern in torch.unique(availability, dim=0):
            observed = [i for i, present in enumerate(pattern.tolist()) if present > 0.5]
            if len(observed) < 2:
                continue
            sample_idx = torch.nonzero((availability == pattern).all(dim=1), as_tuple=False).flatten()
            if sample_idx.numel() < self.min_symile_group_size:
                continue
            sample_count += int(sample_idx.numel())
            rel = reliability[sample_idx][:, observed, rel_idx]
            rel = rel / rel.sum(dim=1, keepdim=True).clamp_min(1e-6)
            reps = []
            for pos, modality_index in enumerate(observed):
                modality = STANDARD_MODALITIES[modality_index]
                projected = self._project(modality, all_heads[modality][space][sample_idx])
                reps.append(projected * rel[:, pos : pos + 1])
            reps.append(self._project(anchor_name, anchor[sample_idx]))
            losses.append(self._mip_shuffle(reps))
            group_count += 1
        if not losses:
            return self._zero_like_reference(reference), 0, 0
        return torch.stack(losses).mean(), group_count, sample_count

    def _anchor_space(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        availability: torch.Tensor,
        reliability: torch.Tensor,
        anchor: torch.Tensor,
        space: str,
        rel_idx: int,
        anchor_name: str,
    ) -> torch.Tensor:
        losses = []
        for modality_idx, modality in enumerate(STANDARD_MODALITIES):
            sample_idx = torch.nonzero(availability[:, modality_idx] > 0.5, as_tuple=False).flatten()
            if sample_idx.numel() == 0:
                continue
            query = self._project(modality, all_heads[modality][space][sample_idx])
            key = self._project(anchor_name, anchor[sample_idx])
            weight = reliability[sample_idx, modality_idx, rel_idx].detach().mean()
            losses.append(weight * _info_nce_or_cosine(query, key, self.temperature))
        if not losses:
            return self._zero_like_reference(anchor)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _anchor_alignment_space(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        availability: torch.Tensor,
        anchor: torch.Tensor,
        space: str,
        anchor_name: str,
    ) -> torch.Tensor:
        alignments = []
        for modality_idx, modality in enumerate(STANDARD_MODALITIES):
            sample_idx = torch.nonzero(availability[:, modality_idx] > 0.5, as_tuple=False).flatten()
            if sample_idx.numel() == 0:
                continue
            query = self._project(modality, all_heads[modality][space][sample_idx])
            key = self._project(anchor_name, anchor[sample_idx])
            alignments.append(F.cosine_similarity(query, key, dim=1, eps=1e-8).mean())
        if not alignments:
            return self._zero_like_reference(anchor)
        return torch.stack(alignments).mean()

    @staticmethod
    def _weighted_pair_loss(
        query: torch.Tensor,
        key: torch.Tensor,
        sample_weight: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        if query.shape[0] < 2:
            per_sample = 1.0 - F.cosine_similarity(query, key, dim=1, eps=1e-8)
        else:
            logits = query @ key.transpose(0, 1) / float(temperature)
            labels = torch.arange(logits.shape[0], device=logits.device)
            query_loss = F.cross_entropy(logits, labels, reduction="none")
            key_loss = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
            per_sample = 0.5 * (query_loss + key_loss)
        return (per_sample * sample_weight).mean()

    @staticmethod
    @torch.no_grad()
    def _retrieval_accuracy(query: torch.Tensor, key: torch.Tensor, top_k: int) -> torch.Tensor:
        if query.shape[0] < 2:
            return query.new_zeros(())
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        logits = query @ key.transpose(0, 1)
        labels = torch.arange(logits.shape[0], device=logits.device)
        k = min(int(top_k), int(logits.shape[1]))
        query_hits = logits.topk(k, dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean()
        key_hits = logits.transpose(0, 1).topk(k, dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean()
        return 0.5 * (query_hits + key_hits)

    @torch.no_grad()
    def _pair_alignment(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if query.shape[0] == 0:
            return query.new_zeros(())
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        return F.cosine_similarity(query, key, dim=1, eps=1e-8).mean()

    def _pairwise_space(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        availability: torch.Tensor,
        reliability: torch.Tensor,
        space: str,
        rel_idx: int,
    ) -> dict[str, torch.Tensor | int]:
        reference = next(iter(next(iter(all_heads.values())).values()))
        losses: list[torch.Tensor] = []
        top1_values: list[torch.Tensor] = []
        top5_values: list[torch.Tensor] = []
        pair_count = 0
        retrieval_pair_count = 0
        special_top5 = self._zero_like_reference(reference)
        special_alignment = self._zero_like_reference(reference)
        special_seen = False

        for left_idx, left in enumerate(STANDARD_MODALITIES):
            for right_idx in range(left_idx + 1, len(STANDARD_MODALITIES)):
                right = STANDARD_MODALITIES[right_idx]
                observed = (availability[:, left_idx] > 0.5) & (availability[:, right_idx] > 0.5)
                sample_idx = torch.nonzero(observed, as_tuple=False).flatten()
                if sample_idx.numel() == 0:
                    continue
                pair_count += 1
                query = self._project(left, all_heads[left][space][sample_idx])
                key = self._project(right, all_heads[right][space][sample_idx])
                pair_weight = torch.minimum(
                    reliability[sample_idx, left_idx, rel_idx],
                    reliability[sample_idx, right_idx, rel_idx],
                ).detach()
                pair_weight = pair_weight.clamp_min(self.pairwise_reliability_floor)
                losses.append(self._weighted_pair_loss(query, key, pair_weight, self.pairwise_temperature))

                is_well_rms_num = (
                    space == "numerical"
                    and {left, right} == {"well_log", "rms_vel"}
                )
                if is_well_rms_num:
                    special_alignment = self._pair_alignment(query, key)
                    special_seen = True
                if sample_idx.numel() < 2:
                    continue
                retrieval_pair_count += 1
                top1 = self._retrieval_accuracy(query, key, 1)
                top5 = self._retrieval_accuracy(query, key, 5)
                top1_values.append(top1)
                top5_values.append(top5)
                if is_well_rms_num:
                    special_top5 = top5

        loss = torch.stack(losses).mean() if losses else self._zero_like_reference(reference)
        top1 = torch.stack(top1_values).mean() if top1_values else self._zero_like_reference(reference)
        top5 = torch.stack(top5_values).mean() if top5_values else self._zero_like_reference(reference)
        if not special_seen:
            special_alignment = self._zero_like_reference(reference)
        return {
            "loss": loss,
            "top1": top1,
            "top5": top5,
            "pair_count": pair_count,
            "retrieval_pair_count": retrieval_pair_count,
            "well_log_rms_vel_top5": special_top5,
            "well_log_rms_vel_alignment": special_alignment,
        }

    def _orthogonality(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sn_losses = []
        unique_losses = []
        for modality_idx, modality in enumerate(STANDARD_MODALITIES):
            sample_idx = torch.nonzero(availability[:, modality_idx] > 0.5, as_tuple=False).flatten()
            if sample_idx.numel() == 0:
                continue
            heads = all_heads[modality]
            s = heads["structural"][sample_idx]
            n = heads["numerical"][sample_idx]
            u = heads["unique"][sample_idx]
            sn_losses.append(self._cos_abs(s, n).mean())
            unique_losses.append(0.5 * (self._cos_abs(u, s).mean() + self._cos_abs(u, n).mean()))
        reference = next(iter(next(iter(all_heads.values())).values()))
        sn = torch.stack(sn_losses).mean() if sn_losses else self._zero_like_reference(reference)
        unique = torch.stack(unique_losses).mean() if unique_losses else self._zero_like_reference(reference)
        return sn, unique

    def _low_high_energy_ratios(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.frequency_kernel_size // 2
        low = F.avg_pool2d(x, kernel_size=self.frequency_kernel_size, stride=1, padding=pad)
        high = x - low
        low_energy = low.square().flatten(1).mean(dim=1)
        high_energy = high.square().flatten(1).mean(dim=1)
        total = (low_energy + high_energy).clamp_min(1e-8)
        return low_energy / total, high_energy / total

    def _frequency_decoupling(
        self,
        all_heads: Mapping[str, Mapping[str, torch.Tensor]],
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        structural_low_penalties = []
        numerical_high_penalties = []
        structural_high_ratios = []
        numerical_low_ratios = []
        for modality_idx, modality in enumerate(STANDARD_MODALITIES):
            sample_idx = torch.nonzero(availability[:, modality_idx] > 0.5, as_tuple=False).flatten()
            if sample_idx.numel() == 0:
                continue
            heads = all_heads[modality]
            s_low, s_high = self._low_high_energy_ratios(heads["structural"][sample_idx])
            n_low, n_high = self._low_high_energy_ratios(heads["numerical"][sample_idx])
            structural_low_penalties.append(s_low.mean())
            numerical_high_penalties.append(n_high.mean())
            structural_high_ratios.append(s_high.mean())
            numerical_low_ratios.append(n_low.mean())

        reference = next(iter(next(iter(all_heads.values())).values()))
        if not structural_low_penalties:
            zero = self._zero_like_reference(reference)
            return zero, zero, zero, zero
        return (
            torch.stack(structural_low_penalties).mean(),
            torch.stack(numerical_high_penalties).mean(),
            torch.stack(structural_high_ratios).mean(),
            torch.stack(numerical_low_ratios).mean(),
        )

    def _reliability_prior_loss(self, reliability: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        prior = self.reliability_prior.to(device=reliability.device, dtype=reliability.dtype)
        diff = (reliability - prior.unsqueeze(0)).square().mean(dim=2)
        observed = availability > 0.5
        if not observed.any():
            return reliability.new_zeros(())
        return diff[observed].mean()

    def forward(
        self,
        features: DecoupledFeatures,
        batch: BGSampleBatch,
        *,
        anchor_struct: torch.Tensor,
        anchor_num: torch.Tensor,
    ) -> dict[str, Any]:
        if features.all_heads is None:
            raise ValueError("PhysicsAnchoredSubsetSymileLoss requires features.all_heads.")
        all_heads = features.all_heads
        modality_mask = batch.modality_mask.to(features.reliability.device)
        modality_quality = batch.modality_quality.to(features.reliability.device)
        availability = (modality_mask * modality_quality).gt(0).to(features.reliability.dtype)
        reliability = features.reliability

        sym_s, groups_s, samples_s = self._symile_space(
            all_heads,
            reliability,
            availability,
            anchor_struct,
            "structural",
            0,
            "anchor_struct",
        )
        sym_n, groups_n, samples_n = self._symile_space(
            all_heads,
            reliability,
            availability,
            anchor_num,
            "numerical",
            1,
            "anchor_num",
        )
        pair_s = self._pairwise_space(all_heads, availability, reliability, "structural", 0)
        pair_n = self._pairwise_space(all_heads, availability, reliability, "numerical", 1)
        anchor_s = self._anchor_space(
            all_heads,
            availability,
            reliability,
            anchor_struct,
            "structural",
            0,
            "anchor_struct",
        )
        anchor_n = self._anchor_space(all_heads, availability, reliability, anchor_num, "numerical", 1, "anchor_num")
        anchor = 0.5 * (anchor_s + anchor_n)
        sn_ortho, unique_ortho = self._orthogonality(all_heads, availability)
        structural_low_penalty, numerical_high_penalty, structural_high_ratio, numerical_low_ratio = (
            self._frequency_decoupling(all_heads, availability)
        )
        reliability_prior = self._reliability_prior_loss(reliability, availability)
        observed_counts = availability.sum(dim=1)
        unique_patterns = torch.unique(availability, dim=0)
        observed_subset_count = int((unique_patterns.sum(dim=1) > 0).sum().item())
        structural_alignment = self._anchor_alignment_space(
            all_heads,
            availability,
            anchor_struct,
            "structural",
            "anchor_struct",
        )
        numerical_alignment = self._anchor_alignment_space(
            all_heads,
            availability,
            anchor_num,
            "numerical",
            "anchor_num",
        )

        total = (
            self.weights["symile_structural"] * sym_s
            + self.weights["symile_numerical"] * sym_n
            + self.weights["pairwise_structural"] * pair_s["loss"]
            + self.weights["pairwise_numerical"] * pair_n["loss"]
            + self.weights["anchor"] * anchor
            + self.weights["reliability_prior"] * reliability_prior
            + self.weights["sn_ortho"] * sn_ortho
            + self.weights["unique_ortho"] * unique_ortho
            + self.weights["structural_highpass"] * structural_low_penalty
            + self.weights["numerical_lowpass"] * numerical_high_penalty
        )
        output: dict[str, Any] = {
            "loss": total,
            "symile_structural": sym_s,
            "symile_numerical": sym_n,
            "pairwise_structural": pair_s["loss"],
            "pairwise_numerical": pair_n["loss"],
            "pairwise_retrieval_top1": 0.5 * (pair_s["top1"] + pair_n["top1"]),
            "pairwise_retrieval_top5": 0.5 * (pair_s["top5"] + pair_n["top5"]),
            "pairwise_retrieval_top1_structural": pair_s["top1"],
            "pairwise_retrieval_top5_structural": pair_s["top5"],
            "pairwise_retrieval_top1_numerical": pair_n["top1"],
            "pairwise_retrieval_top5_numerical": pair_n["top5"],
            "pairwise_retrieval_top5_numerical_well_log_rms_vel": pair_n["well_log_rms_vel_top5"],
            "pairwise_alignment_numerical_well_log_rms_vel": pair_n["well_log_rms_vel_alignment"],
            "pairwise_structural_pairs": pair_s["pair_count"],
            "pairwise_numerical_pairs": pair_n["pair_count"],
            "pairwise_structural_retrieval_pairs": pair_s["retrieval_pair_count"],
            "pairwise_numerical_retrieval_pairs": pair_n["retrieval_pair_count"],
            "anchor": anchor,
            "anchor_structural": anchor_s,
            "anchor_numerical": anchor_n,
            "reliability_prior": reliability_prior,
            "sn_ortho": sn_ortho,
            "unique_ortho": unique_ortho,
            "frequency_structural_low_penalty": structural_low_penalty,
            "frequency_numerical_high_penalty": numerical_high_penalty,
            "frequency_structural_high_ratio": structural_high_ratio,
            "frequency_numerical_low_ratio": numerical_low_ratio,
            "symile_structural_groups": groups_s,
            "symile_numerical_groups": groups_n,
            "symile_structural_samples": samples_s,
            "symile_numerical_samples": samples_n,
            "observed_subset_count": observed_subset_count,
            "observed_modality_mean": observed_counts.mean(),
            "observed_modality_min": observed_counts.min(),
            "observed_modality_max": observed_counts.max(),
            "structural_anchor_alignment": structural_alignment,
            "numerical_anchor_alignment": numerical_alignment,
        }
        for modality_idx, modality in enumerate(STANDARD_MODALITIES):
            observed = availability[:, modality_idx] > 0.5
            if observed.any():
                rel = reliability[observed, modality_idx]
                output[f"reliability_{modality}_structural"] = rel[:, 0].mean()
                output[f"reliability_{modality}_numerical"] = rel[:, 1].mean()
            else:
                output[f"reliability_{modality}_structural"] = reliability.new_zeros(())
                output[f"reliability_{modality}_numerical"] = reliability.new_zeros(())
        return output


def build_contrastive_loss(conf: Any, feature_channels: int) -> PhysicsAnchoredSubsetSymileLoss:
    def get(path: str, default: Any) -> Any:
        current = conf
        for part in path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
            else:
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
        return current

    return PhysicsAnchoredSubsetSymileLoss(
        feature_channels=feature_channels,
        embed_dim=int(get("contrastive.embed_dim", 128)),
        hidden_dim=get("contrastive.hidden_dim", None),
        temperature=float(get("contrastive.temperature", 0.1)),
        pairwise_temperature=float(get("contrastive.pairwise_temperature", 0.1)),
        symile_structural_weight=float(get("contrastive.symile_structural_weight", 1.0)),
        symile_numerical_weight=float(get("contrastive.symile_numerical_weight", 1.0)),
        pairwise_structural_weight=float(get("contrastive.pairwise_structural_weight", 0.0)),
        pairwise_numerical_weight=float(get("contrastive.pairwise_numerical_weight", 0.0)),
        pairwise_reliability_floor=float(get("contrastive.pairwise_reliability_floor", 0.35)),
        anchor_weight=float(get("contrastive.anchor_weight", 0.5)),
        reliability_prior_weight=float(get("contrastive.reliability_prior_weight", 0.05)),
        sn_ortho_weight=float(get("contrastive.sn_ortho_weight", 0.05)),
        unique_ortho_weight=float(get("contrastive.unique_ortho_weight", 0.05)),
        structural_highpass_weight=float(get("contrastive.structural_highpass_weight", 0.0)),
        numerical_lowpass_weight=float(get("contrastive.numerical_lowpass_weight", 0.0)),
        frequency_kernel_size=int(get("contrastive.frequency_kernel_size", 5)),
        min_symile_group_size=int(get("contrastive.min_symile_group_size", 2)),
        strict_derangement=bool(get("contrastive.strict_derangement", True)),
        num_negative_shuffles=get("contrastive.num_negative_shuffles", None),
    )


def build_wavelet_anchor(conf: Any) -> DepthVelocityWaveletTransform:
    def get(path: str, default: Any) -> Any:
        current = conf
        for part in path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
            else:
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
        return current

    return DepthVelocityWaveletTransform(
        level=int(get("contrastive.wavelet.level", 1)),
        boundary=str(get("contrastive.wavelet.boundary", "edge")),
        resize_mode=str(get("contrastive.wavelet.resize_mode", "nearest")),
        detail_fusion=str(get("contrastive.wavelet.detail_fusion", "l2")),
    )

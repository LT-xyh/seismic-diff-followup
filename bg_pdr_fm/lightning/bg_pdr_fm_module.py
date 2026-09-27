"""Lightning module that coordinates BG-PDR-FM contrastive/background/residual stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import lightning
import torch

from bg_pdr_fm.data.batch import (
    BGInferenceBatch,
    BGSampleBatch,
    STANDARD_MODALITIES,
    collate_bg_samples,
    inference_batch_from_mapping,
)
from bg_pdr_fm.diagnostics import append_metrics_csv, save_metric_pair_plot, save_tensor_panel
from bg_pdr_fm.lightning.stage_losses import (
    background_stage_loss,
    conf_get,
    contrastive_stage_loss,
    joint_full_stage_loss,
    residual_stage_loss,
    true_residual_low_frequency_ratio,
)
from bg_pdr_fm.models import (
    BackgroundEstimator,
    ConditionAdapters,
    LowHighPassFilter,
    PhysicsDecoupledEncoder,
    PredictionBatch,
    ResidualFlowGenerator,
    RhoCalibrator,
    build_contrastive_loss,
    build_latent_codec,
    build_wavelet_anchor,
)


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    return conf_get(conf, path, default)


def _metric_float(value: object) -> float | str:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return ""


def _feature_low_high_ratios(filter_module: LowHighPassFilter, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = filter_module(x.detach())
    low_energy = low.abs().flatten(1).mean(dim=1)
    high_energy = high.abs().flatten(1).mean(dim=1)
    total = (low_energy + high_energy).clamp_min(1e-6)
    return low_energy / total, high_energy / total


class BGPDRFMLightning(lightning.LightningModule):
    """Clean first-milestone BG-PDR-FM training skeleton."""

    def __init__(self, conf: Any | None = None) -> None:
        super().__init__()
        self.conf = conf or {}
        self.stage = str(_conf_get(self.conf, "training.stage", "background"))
        self.lr = float(_conf_get(self.conf, "training.lr", 1e-4))
        self.freeze_encoder = bool(_conf_get(self.conf, "training.freeze_encoder", False))
        self.fusion_mode = str(_conf_get(self.conf, "model.fusion_mode", "reliability")).lower()
        self.save_stage_outputs = bool(_conf_get(self.conf, "training.save_stage_outputs", True))
        self.stage_checkpoint_path = Path(str(_conf_get(
            self.conf, "training.stage_checkpoint_path", "logs/bg_pdr_fm/stage_checkpoints"
        )))
        self.load_stage_checkpoint = _conf_get(self.conf, "training.load_stage_checkpoint", None)
        self.stage_checkpoint_used = ""
        self.diagnostics_enabled = bool(_conf_get(self.conf, "diagnostics.enabled", True))
        self.diagnostics_output_dir = Path(
            str(_conf_get(self.conf, "diagnostics.output_dir", "logs/bg_pdr_fm/diagnostics"))
        )
        self.contrastive_diagnostics_frequency = bool(
            _conf_get(self.conf, "contrastive.diagnostics.frequency_metrics", True)
        )
        self.contrastive_diagnostics_reliability = bool(
            _conf_get(self.conf, "contrastive.diagnostics.reliability_metrics", True)
        )
        self.contrastive_diagnostics_subset = bool(
            _conf_get(self.conf, "contrastive.diagnostics.subset_metrics", True)
        )
        self.modality_dropout_enabled = bool(_conf_get(self.conf, "contrastive.modality_dropout.enabled", False))
        self.modality_dropout_probs = torch.tensor(
            [
                float(_conf_get(self.conf, f"contrastive.modality_dropout.probs.{name}", 0.0))
                for name in STANDARD_MODALITIES
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        self._diagnostic_written: set[str] = set()
        feature_channels = int(_conf_get(self.conf, "model.feature_channels", 32))
        self.uses_generation_stack = self.stage != "contrastive"
        cond_channels = int(_conf_get(self.conf, "model.cond_channels", 64))
        latent_channels = int(_conf_get(self.conf, "model.latent_channels", 16))
        latent_hw = tuple(_conf_get(self.conf, "model.latent_hw", (16, 16)))
        self.codec_type = str(_conf_get(self.conf, "model.codec_type", "simple"))
        self.residual_backend = str(_conf_get(self.conf, "model.residual_backend", "simple_fm"))
        self.residual_num_inference_steps = int(_conf_get(self.conf, "model.residual_num_inference_steps", 20))
        self.quality_gate_mode = str(_conf_get(self.conf, "quality_gate.mode", "none")).lower()
        self.background_condition_source = str(
            _conf_get(self.conf, "model.background_condition_source", "adapter_numerical")
        ).lower()
        if self.background_condition_source not in {
            "adapter_numerical",
            "encoder_numerical",
            "encoder_numerical_raw_bypass",
        }:
            raise ValueError(
                "model.background_condition_source must be one of 'adapter_numerical', 'encoder_numerical', or "
                f"'encoder_numerical_raw_bypass', got {self.background_condition_source!r}."
            )
        self.background_raw_bypass_modalities = tuple(
            _conf_get(self.conf, "model.background_raw_bypass_modalities", ("rms_vel",))
        )
        self.background_raw_bypass_channels = int(
            _conf_get(self.conf, "model.background_raw_bypass_channels", feature_channels)
        )

        self.filter = LowHighPassFilter(kernel_size=int(_conf_get(self.conf, "model.lowpass_kernel", 5)))
        self.wavelet_anchor = build_wavelet_anchor(self.conf)
        self.encoder = PhysicsDecoupledEncoder(
            hidden_channels=int(_conf_get(self.conf, "model.hidden_channels", 32)),
            base_channels=int(_conf_get(self.conf, "model.base_channels", 64)),
            feature_channels=feature_channels,
            fusion_mode=self.fusion_mode,
        )
        if self.uses_generation_stack:
            self.codec = build_latent_codec(
                codec_type=self.codec_type,
                latent_channels=latent_channels,
                latent_hw=latent_hw,
                checkpoint_path=_conf_get(self.conf, "model.codec_checkpoint", None),
                module_path=_conf_get(self.conf, "model.codec_module_path", None),
            )
            self.adapters = ConditionAdapters(
                feature_channels=feature_channels,
                cond_channels=cond_channels,
                out_hw=latent_hw,
                use_unique=bool(_conf_get(self.conf, "model.use_unique", False)),
                unique_eta=float(_conf_get(self.conf, "model.unique_eta", 0.1)),
            )
            self.background_raw_bypass = None
            if self.background_condition_source == "encoder_numerical_raw_bypass":
                self.background_raw_bypass = torch.nn.Sequential(
                    torch.nn.Conv2d(len(self.background_raw_bypass_modalities), self.background_raw_bypass_channels,
                                    kernel_size=3, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(self.background_raw_bypass_channels, self.background_raw_bypass_channels,
                                    kernel_size=3, padding=1),
                    torch.nn.SiLU(),
                )
            background_cond_channels = cond_channels
            if self.background_condition_source == "encoder_numerical":
                background_cond_channels = feature_channels
            elif self.background_condition_source == "encoder_numerical_raw_bypass":
                background_cond_channels = feature_channels + self.background_raw_bypass_channels
            self.background = BackgroundEstimator(
                cond_channels=background_cond_channels,
                latent_channels=latent_channels,
                highpass_filter=self.filter,
                output_activation=str(_conf_get(self.conf, "model.background_output_activation", "tanh")),
                hidden_channels=int(
                    _conf_get(
                        self.conf,
                        "model.background_fm_hidden_channels",
                        _conf_get(self.conf, "model.background_hidden_channels", cond_channels),
                    )
                    if str(_conf_get(self.conf, "model.background_backend", "conv")).lower() == "flow_matching"
                    else _conf_get(self.conf, "model.background_hidden_channels", cond_channels)
                ),
                output_mode=str(_conf_get(self.conf, "model.background_output_mode", "direct")),
                wavelet_level=int(_conf_get(self.conf, "model.background_wavelet_level", 1)),
                bottleneck_hw=tuple(_conf_get(self.conf, "model.background_bottleneck_hw", (18, 18))),
                upsample_mode=str(_conf_get(self.conf, "model.background_upsample_mode", "bilinear")),
                backend=str(_conf_get(self.conf, "model.background_backend", "conv")),
                fm_backend=str(_conf_get(self.conf, "model.background_fm_backend", "unet_fm_film")),
                fm_latent_hw=tuple(_conf_get(self.conf, "model.background_fm_latent_hw", (70, 70))),
                fm_num_train_timesteps=int(_conf_get(self.conf, "model.background_fm_num_train_timesteps", 1000)),
                fm_num_inference_steps=int(_conf_get(self.conf, "model.background_fm_num_inference_steps", 20)),
                fm_loss_type=str(_conf_get(self.conf, "model.background_fm_loss_type", "mse")),
                fm_block_out_channels=_conf_get(self.conf, "model.background_fm_block_out_channels", None),
                fm_layers_per_block=int(_conf_get(self.conf, "model.background_fm_layers_per_block", 2)),
                fm_cross_attention_dim=_conf_get(self.conf, "model.background_fm_cross_attention_dim", None),
                fm_attention_head_dim=int(_conf_get(self.conf, "model.background_fm_attention_head_dim", 8)),
            )
            self.residual = ResidualFlowGenerator(
                cond_channels=cond_channels,
                latent_channels=latent_channels,
                latent_hw=latent_hw,
                backend=self.residual_backend,
                num_train_timesteps=int(_conf_get(self.conf, "model.residual_num_train_timesteps", 1000)),
                num_inference_steps=self.residual_num_inference_steps,
                loss_type=str(_conf_get(self.conf, "model.residual_loss_type", "mse")),
                lowpass_filter=self.filter,
                backend_hidden_channels=int(_conf_get(self.conf, "model.residual_backend_hidden_channels", 96)),
                condition_merge=str(_conf_get(self.conf, "model.residual_condition_merge", "add")),
                background_context_source=str(
                    _conf_get(self.conf, "model.residual_background_context_source", "lowpass_embed")
                ),
                backend_block_out_channels=_conf_get(self.conf, "model.residual_block_out_channels", None),
                backend_layers_per_block=int(_conf_get(self.conf, "model.residual_layers_per_block", 2)),
                backend_cross_attention_dim=_conf_get(self.conf, "model.residual_cross_attention_dim", None),
                backend_attention_head_dim=int(_conf_get(self.conf, "model.residual_attention_head_dim", 8)),
            )
            self.rho_calibrator = RhoCalibrator(tau=float(_conf_get(self.conf, "model.rho_tau", 1.0)))
        else:
            self.codec = None
            self.adapters = None
            self.background = None
            self.residual = None
            self.rho_calibrator = None
        self.num_to_anchor = torch.nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.struct_to_anchor = torch.nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.contrastive_loss = build_contrastive_loss(self.conf, feature_channels=feature_channels)
        self._last_contrastive_metrics: dict[str, object] = {}
        self._apply_stage_trainability()
        self._load_stage_checkpoint_if_requested()

    def _set_requires_grad(self, module: torch.nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _apply_stage_trainability(self) -> None:
        self._set_requires_grad(self.encoder, self.stage in {"contrastive", "joint_full"})
        self._set_requires_grad(self.num_to_anchor, True)
        self._set_requires_grad(self.struct_to_anchor, True)
        self._set_requires_grad(self.contrastive_loss, True)

        if self.stage == "contrastive":
            self._set_requires_grad(self.num_to_anchor, False)
            self._set_requires_grad(self.struct_to_anchor, False)
        else:
            assert self.adapters is not None
            assert self.rho_calibrator is not None
            assert self.background is not None
            assert self.residual is not None
            assert self.codec is not None
            self._set_requires_grad(self.adapters, True)
            train_gate = self.quality_gate_mode == "learned"
            self._set_requires_grad(self.rho_calibrator, train_gate)
            self._set_requires_grad(self.background, True)
            if self.background_raw_bypass is not None:
                self._set_requires_grad(self.background_raw_bypass, True)
            self._set_requires_grad(self.residual, True)
            self._set_requires_grad(self.codec, False)
            self._set_requires_grad(self.contrastive_loss, self.stage == "joint_full")
            self._set_requires_grad(self.encoder, self.stage == "joint_full")
            self._set_requires_grad(self.num_to_anchor, False)
            self._set_requires_grad(self.struct_to_anchor, False)
            if self.stage == "residual":
                self._set_requires_grad(self.adapters.numerical, False)
                self._set_requires_grad(self.adapters.unique, False)
                self._set_requires_grad(self.background, False)
            if self.stage == "background":
                self._set_requires_grad(self.adapters.structural, False)
                self._set_requires_grad(self.adapters.unique, False)
                if self.background_condition_source in {"encoder_numerical", "encoder_numerical_raw_bypass"}:
                    self._set_requires_grad(self.adapters.numerical, False)
                self._set_requires_grad(self.residual, False)
                if bool(_conf_get(self.conf, "training.tune_encoder_numerical_heads_for_background", False)):
                    self._set_requires_grad(self.encoder, False)
                    for head in self.encoder.heads.values():
                        self._set_requires_grad(head.numerical, True)
            if self.stage == "joint_full" and self.background_condition_source in {
                "encoder_numerical",
                "encoder_numerical_raw_bypass",
            }:
                self._set_requires_grad(self.adapters.numerical, False)
                self._set_requires_grad(self.adapters.unique, False)
            if self.stage == "joint_full":
                self._set_requires_grad(self.num_to_anchor, False)
                self._set_requires_grad(self.struct_to_anchor, False)
                if hasattr(self.encoder, "base_heads"):
                    self._set_requires_grad(self.encoder.base_heads, False)
                if self.quality_gate_mode != "learned" and float(_conf_get(self.conf, "loss.rho_weight", 0.0)) == 0.0:
                    self._set_requires_grad(self.rho_calibrator, False)

    @staticmethod
    def _assert_frozen(module: torch.nn.Module, label: str) -> None:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError(f"{label} must be frozen in residual stage.")

    def _assert_stage_trainability(self) -> None:
        if not self.uses_generation_stack:
            return
        assert self.codec is not None
        assert self.background is not None
        assert self.adapters is not None
        assert self.rho_calibrator is not None
        if self.stage == "residual":
            self._assert_frozen(self.encoder, "encoder")
            self._assert_frozen(self.codec, "codec")
            self._assert_frozen(self.adapters.numerical, "numerical adapter")
            self._assert_frozen(self.background, "background estimator")
            if self.quality_gate_mode != "learned":
                self._assert_frozen(self.rho_calibrator, "rho calibrator")
            self._assert_frozen(self.num_to_anchor, "numerical anchor head")
            self._assert_frozen(self.struct_to_anchor, "structural anchor head")
        if self.stage == "background" and self.background_condition_source in {
            "encoder_numerical",
            "encoder_numerical_raw_bypass",
        }:
            tune_num_heads = bool(_conf_get(self.conf, "training.tune_encoder_numerical_heads_for_background", False))
            if not tune_num_heads:
                self._assert_frozen(self.encoder, "encoder")
            self._assert_frozen(self.adapters, "condition adapters")
            self._assert_frozen(self.residual, "residual generator")
            self._assert_frozen(self.codec, "codec")
            if tune_num_heads:
                for name, parameter in self.encoder.named_parameters():
                    if name.startswith("heads.") and ".numerical." in name:
                        continue
                    if parameter.requires_grad:
                        raise AssertionError(f"Only numerical encoder heads may be trainable, got {name}.")

    def on_train_epoch_start(self) -> None:
        self._apply_stage_trainability()
        self._assert_stage_trainability()
        if self.uses_generation_stack:
            assert self.codec is not None
            if self.stage != "joint_full":
                self.encoder.eval()
            self.codec.eval()
        if self.stage == "residual":
            assert self.background is not None
            assert self.adapters is not None
            self.background.eval()
            self.adapters.numerical.eval()
        if self.stage == "background" and self.background_condition_source in {
            "encoder_numerical",
            "encoder_numerical_raw_bypass",
        }:
            assert self.adapters is not None
            self.adapters.eval()

    def _prepare_frozen_modules_for_step(self) -> None:
        if not self.uses_generation_stack:
            return
        self._assert_stage_trainability()
        assert self.codec is not None
        if self.stage != "joint_full":
            self.encoder.eval()
        self.codec.eval()
        if self.stage == "residual":
            assert self.background is not None
            assert self.adapters is not None
            self.background.eval()
            self.adapters.numerical.eval()
        if self.stage == "background" and self.background_condition_source in {
            "encoder_numerical",
            "encoder_numerical_raw_bypass",
        }:
            assert self.adapters is not None
            self.adapters.eval()

    def _load_stage_checkpoint_if_requested(self) -> None:
        if self.load_stage_checkpoint is None or str(self.load_stage_checkpoint).strip() == "":
            return
        path = Path(str(self.load_stage_checkpoint))
        if not path.is_file():
            raise FileNotFoundError(f"BG-PDR-FM stage checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        if "state_dict" not in state:
            raise KeyError(f"BG-PDR-FM stage checkpoint {path} does not contain `state_dict`.")
        current_state = self.state_dict()
        compatible_state = {
            key: value
            for key, value in state["state_dict"].items()
            if key not in current_state or tuple(value.shape) == tuple(current_state[key].shape)
        }
        self.load_state_dict(compatible_state, strict=False)
        self.stage_checkpoint_used = str(path)
        if self.stage == "joint_full":
            self._load_joint_warm_start_checkpoints()

    def _load_compatible_checkpoint(self, path: str | Path, label: str) -> None:
        checkpoint_path = Path(str(path))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Joint warm-start {label} checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" not in state:
            raise KeyError(f"Joint warm-start {label} checkpoint {checkpoint_path} does not contain `state_dict`.")
        current_state = self.state_dict()
        compatible_state = {
            key: value
            for key, value in state["state_dict"].items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }
        self.load_state_dict(compatible_state, strict=False)
        used = getattr(self, "joint_warm_start_used", {})
        used[label] = {
            "path": str(checkpoint_path),
            "loaded_keys": len(compatible_state),
            "skipped_keys": len(state["state_dict"]) - len(compatible_state),
        }
        self.joint_warm_start_used = used

    def _load_joint_warm_start_checkpoints(self) -> None:
        warm = _conf_get(self.conf, "training.joint.warm_start_checkpoints", None)
        if warm is None:
            return
        for label in ("contrastive", "background", "residual"):
            path = _conf_get(self.conf, f"training.joint.warm_start_checkpoints.{label}", None)
            if path is None or str(path).strip() == "":
                continue
            self._load_compatible_checkpoint(path, label)

    def _is_global_zero(self) -> bool:
        try:
            trainer = self.trainer
        except RuntimeError:
            return True
        return bool(getattr(trainer, "is_global_zero", True))

    def _sync_dist(self) -> bool:
        try:
            trainer = self.trainer
        except RuntimeError:
            return False
        return int(getattr(trainer, "world_size", 1)) > 1

    def _maybe_apply_modality_dropout(self, batch: BGSampleBatch, prefix: str) -> BGSampleBatch:
        if self.stage != "contrastive" or prefix != "train" or not self.training or not self.modality_dropout_enabled:
            return batch
        device = batch.modality_mask.device
        probs = self.modality_dropout_probs.to(device=device, dtype=batch.modality_mask.dtype).view(1, -1)
        if torch.all(probs <= 0):
            return batch
        original_available = (batch.modality_mask * batch.modality_quality).gt(0)
        keep = torch.rand_like(batch.modality_mask) >= probs
        new_mask = batch.modality_mask * keep.to(batch.modality_mask.dtype)
        new_quality = batch.modality_quality * keep.to(batch.modality_quality.dtype)

        new_available = (new_mask * new_quality).gt(0)
        empty_rows = original_available.any(dim=1) & (~new_available.any(dim=1))
        if empty_rows.any():
            restore_idx = original_available[empty_rows].float().argmax(dim=1)
            row_idx = torch.nonzero(empty_rows, as_tuple=False).flatten()
            new_mask[row_idx, restore_idx] = batch.modality_mask[row_idx, restore_idx]
            new_quality[row_idx, restore_idx] = batch.modality_quality[row_idx, restore_idx]

        return BGSampleBatch(
            depth_vel=batch.depth_vel,
            migrated_image=batch.migrated_image,
            horizon=batch.horizon,
            rms_vel=batch.rms_vel,
            well_log=batch.well_log,
            well_mask=batch.well_mask,
            modality_mask=new_mask,
            modality_quality=new_quality,
            metadata=batch.metadata,
        )

    def _save_stage_checkpoint(self) -> None:
        if not self.save_stage_outputs or not self._is_global_zero():
            return
        self.stage_checkpoint_path.mkdir(parents=True, exist_ok=True)
        output_path = self.stage_checkpoint_path / f"{self.stage}_last.ckpt"
        state = {
            "stage": self.stage,
            "state_dict": self.state_dict(),
        }
        if self.uses_generation_stack:
            state["codec_type"] = self.codec_type
            state["residual_backend"] = self.residual_backend
        torch.save(state, output_path)
        if self.stage == "joint_full":
            torch.save(state, self.stage_checkpoint_path / "contrastive_last.ckpt")
            torch.save(state, self.stage_checkpoint_path / "background_last.ckpt")
            torch.save(state, self.stage_checkpoint_path / "residual_last.ckpt")

    @staticmethod
    def _as_batch(batch: BGSampleBatch | dict[str, torch.Tensor]) -> BGSampleBatch:
        if isinstance(batch, BGSampleBatch):
            return batch
        return collate_bg_samples([batch]) if batch["depth_vel"].ndim == 3 else BGSampleBatch(
            depth_vel=batch["depth_vel"],
            migrated_image=batch["migrated_image"],
            horizon=batch["horizon"],
            rms_vel=batch["rms_vel"],
            well_log=batch["well_log"],
            well_mask=batch["well_mask"],
            modality_mask=batch["modality_mask"],
            modality_quality=batch["modality_quality"],
            metadata={k: v for k, v in batch.items() if k not in {
                "depth_vel", "migrated_image", "horizon", "rms_vel", "well_log", "well_mask",
                "modality_mask", "modality_quality"
            }},
        )

    @staticmethod
    def _as_inference_batch(
        batch: BGInferenceBatch | BGSampleBatch | Mapping[str, torch.Tensor],
    ) -> BGInferenceBatch:
        if isinstance(batch, BGInferenceBatch):
            return batch
        if isinstance(batch, BGSampleBatch):
            return batch.as_inference_batch()
        if isinstance(batch, Mapping):
            return inference_batch_from_mapping(batch)
        raise TypeError(
            "BGPDRFMLightning.predict_batch expects BGInferenceBatch, BGSampleBatch, or an observed-only mapping."
        )

    def _anchors(self, depth_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.filter(depth_vel)

    anchors = _anchors

    def _rho_true(self, depth_vel: torch.Tensor, bg_hat: torch.Tensor) -> torch.Tensor:
        return true_residual_low_frequency_ratio(self.filter, depth_vel, bg_hat)

    def _raw_background_bypass(
        self,
        batch: BGSampleBatch | BGInferenceBatch,
        out_hw: tuple[int, int],
    ) -> torch.Tensor:
        if self.background_raw_bypass is None:
            raise RuntimeError("Raw background bypass is not initialized.")
        tensors: list[torch.Tensor] = []
        for modality in self.background_raw_bypass_modalities:
            if not hasattr(batch, modality):
                raise ValueError(f"Unknown raw bypass modality: {modality!r}.")
            x = getattr(batch, modality)
            if tuple(x.shape[-2:]) != tuple(out_hw):
                x = torch.nn.functional.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
            tensors.append(x)
        raw = torch.cat(tensors, dim=1)
        return self.background_raw_bypass(raw)

    def _background_condition(self, batch: BGSampleBatch | BGInferenceBatch, features, conditions) -> torch.Tensor:
        if self.background_condition_source == "encoder_numerical":
            return features.numerical
        if self.background_condition_source == "encoder_numerical_raw_bypass":
            raw = self._raw_background_bypass(batch, out_hw=batch.output_hw)
            return torch.cat([features.numerical, raw], dim=1)
        assert conditions is not None
        return conditions.numerical

    def _common_forward(self, batch: BGSampleBatch | BGInferenceBatch, *, use_background_target: bool = True):
        features = self.encoder(batch, return_all_heads=self.stage in {"contrastive", "joint_full"})
        if self.stage == "contrastive":
            return features, None, None, None
        assert self.adapters is not None
        assert self.background is not None
        assert self.rho_calibrator is not None
        if self.stage == "background" and self.background_condition_source in {
            "encoder_numerical",
            "encoder_numerical_raw_bypass",
        }:
            conditions = None
        else:
            conditions = self.adapters(features)
        background_condition = self._background_condition(batch, features, conditions)
        background_target = None
        if use_background_target and str(_conf_get(self.conf, "model.background_backend", "conv")).lower() == "flow_matching":
            if not isinstance(batch, BGSampleBatch):
                raise TypeError("Background targets are available only for training batches.")
            bg_target = str(_conf_get(self.conf, "loss.bg_target", "lowpass")).lower()
            if bg_target != "lowpass":
                raise ValueError(f"loss.bg_target must be 'lowpass' for flow-matching background, got {bg_target!r}.")
            background_target = self.filter.lowpass(batch.depth_vel).detach()
        bg = self.background(
            background_condition,
            out_hw=batch.output_hw,
            target=background_target,
        )
        gate = self.rho_calibrator(features.reliability, batch, bg.epsilon_h_b)
        return features, conditions, bg, gate

    def _step(self, batch: BGSampleBatch | dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        self._prepare_frozen_modules_for_step()
        batch = self._as_batch(batch)
        batch = self._maybe_apply_modality_dropout(batch, prefix)
        features, conditions, bg, gate = self._common_forward(batch, use_background_target=True)
        if self.stage == "contrastive":
            stage_output = contrastive_stage_loss(self, batch, features)
        elif self.stage == "background":
            assert bg is not None and gate is not None
            stage_output = background_stage_loss(self, batch, bg, gate)
        elif self.stage == "residual":
            assert conditions is not None and bg is not None and gate is not None
            stage_output = residual_stage_loss(self, batch, conditions, bg, gate)
        elif self.stage == "joint_full":
            assert conditions is not None and bg is not None and gate is not None
            stage_output = joint_full_stage_loss(self, batch, features, conditions, bg, gate)
        else:
            raise ValueError(f"Unknown training.stage: {self.stage}")
        loss = stage_output.loss
        rho_true = stage_output.rho_true
        stage_metrics = stage_output.metrics or {}

        sync_dist = self._sync_dist()
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=sync_dist)
        if self.stage == "contrastive":
            self._log_contrastive_metrics(prefix, sync_dist=sync_dist)
        else:
            assert bg is not None and gate is not None and rho_true is not None
            self.log(f"{prefix}/epsilon_h_b", bg.epsilon_h_b.mean(), on_step=False, on_epoch=True,
                     sync_dist=sync_dist)
            self.log(f"{prefix}/rho_true", rho_true.mean(), on_step=False, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{prefix}/rho_hat_b", gate.rho_hat_b.mean(), on_step=False, on_epoch=True,
                     sync_dist=sync_dist)
            self.log(f"{prefix}/alpha_hat", gate.alpha_hat.mean(), on_step=False, on_epoch=True,
                     sync_dist=sync_dist)
            for metric_name, metric_value in stage_metrics.items():
                self.log(
                    f"{prefix}/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=sync_dist,
                )
        self._write_diagnostics(prefix, batch, features, bg, gate, rho_true, loss, stage_metrics)
        return loss

    def _log_contrastive_metrics(self, prefix: str, sync_dist: bool = False) -> None:
        for name in (
            "symile_structural",
            "symile_numerical",
            "pairwise_structural",
            "pairwise_numerical",
            "pairwise_retrieval_top1",
            "pairwise_retrieval_top5",
            "pairwise_retrieval_top1_structural",
            "pairwise_retrieval_top5_structural",
            "pairwise_retrieval_top1_numerical",
            "pairwise_retrieval_top5_numerical",
            "pairwise_retrieval_top5_numerical_well_log_rms_vel",
            "pairwise_alignment_numerical_well_log_rms_vel",
            "anchor",
            "reliability_prior",
            "sn_ortho",
            "unique_ortho",
            "frequency_structural_low_penalty",
            "frequency_numerical_high_penalty",
            "frequency_structural_high_ratio",
            "frequency_numerical_low_ratio",
            "structural_anchor_alignment",
            "numerical_anchor_alignment",
            "observed_modality_mean",
        ):
            value = self._last_contrastive_metrics.get(name)
            if torch.is_tensor(value):
                self.log(f"{prefix}/{name}", value.detach(), on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=sync_dist)
        for modality in STANDARD_MODALITIES:
            for space in ("structural", "numerical"):
                value = self._last_contrastive_metrics.get(f"reliability_{modality}_{space}")
                if torch.is_tensor(value):
                    self.log(f"{prefix}/reliability_{modality}_{space}", value.detach(), on_step=False,
                             on_epoch=True, sync_dist=sync_dist)

    def _contrastive_metric_row(self, features) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "symile_structural": _metric_float(self._last_contrastive_metrics.get("symile_structural")),
            "symile_numerical": _metric_float(self._last_contrastive_metrics.get("symile_numerical")),
            "pairwise_structural": _metric_float(self._last_contrastive_metrics.get("pairwise_structural")),
            "pairwise_numerical": _metric_float(self._last_contrastive_metrics.get("pairwise_numerical")),
            "pairwise_retrieval_top1": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top1")
            ),
            "pairwise_retrieval_top5": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top5")
            ),
            "pairwise_retrieval_top1_structural": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top1_structural")
            ),
            "pairwise_retrieval_top5_structural": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top5_structural")
            ),
            "pairwise_retrieval_top1_numerical": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top1_numerical")
            ),
            "pairwise_retrieval_top5_numerical": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top5_numerical")
            ),
            "pairwise_retrieval_top5_numerical_well_log_rms_vel": _metric_float(
                self._last_contrastive_metrics.get("pairwise_retrieval_top5_numerical_well_log_rms_vel")
            ),
            "pairwise_alignment_numerical_well_log_rms_vel": _metric_float(
                self._last_contrastive_metrics.get("pairwise_alignment_numerical_well_log_rms_vel")
            ),
            "anchor": _metric_float(self._last_contrastive_metrics.get("anchor")),
            "reliability_prior": _metric_float(self._last_contrastive_metrics.get("reliability_prior")),
            "sn_ortho": _metric_float(self._last_contrastive_metrics.get("sn_ortho")),
            "unique_ortho": _metric_float(self._last_contrastive_metrics.get("unique_ortho")),
            "frequency_structural_low_penalty": _metric_float(
                self._last_contrastive_metrics.get("frequency_structural_low_penalty")
            ),
            "frequency_numerical_high_penalty": _metric_float(
                self._last_contrastive_metrics.get("frequency_numerical_high_penalty")
            ),
            "frequency_structural_high_ratio": _metric_float(
                self._last_contrastive_metrics.get("frequency_structural_high_ratio")
            ),
            "frequency_numerical_low_ratio": _metric_float(
                self._last_contrastive_metrics.get("frequency_numerical_low_ratio")
            ),
            "structural_anchor_alignment": _metric_float(
                self._last_contrastive_metrics.get("structural_anchor_alignment")
            ),
            "numerical_anchor_alignment": _metric_float(
                self._last_contrastive_metrics.get("numerical_anchor_alignment")
            ),
        }
        if self.contrastive_diagnostics_subset:
            row.update(
                {
                    "symile_structural_groups": _metric_float(
                        self._last_contrastive_metrics.get("symile_structural_groups")
                    ),
                    "symile_numerical_groups": _metric_float(
                        self._last_contrastive_metrics.get("symile_numerical_groups")
                    ),
                    "symile_structural_samples": _metric_float(
                        self._last_contrastive_metrics.get("symile_structural_samples")
                    ),
                    "symile_numerical_samples": _metric_float(
                        self._last_contrastive_metrics.get("symile_numerical_samples")
                    ),
                    "pairwise_structural_pairs": _metric_float(
                        self._last_contrastive_metrics.get("pairwise_structural_pairs")
                    ),
                    "pairwise_numerical_pairs": _metric_float(
                        self._last_contrastive_metrics.get("pairwise_numerical_pairs")
                    ),
                    "pairwise_structural_retrieval_pairs": _metric_float(
                        self._last_contrastive_metrics.get("pairwise_structural_retrieval_pairs")
                    ),
                    "pairwise_numerical_retrieval_pairs": _metric_float(
                        self._last_contrastive_metrics.get("pairwise_numerical_retrieval_pairs")
                    ),
                    "observed_subset_count": _metric_float(
                        self._last_contrastive_metrics.get("observed_subset_count")
                    ),
                    "observed_modality_mean": _metric_float(
                        self._last_contrastive_metrics.get("observed_modality_mean")
                    ),
                    "observed_modality_min": _metric_float(
                        self._last_contrastive_metrics.get("observed_modality_min")
                    ),
                    "observed_modality_max": _metric_float(
                        self._last_contrastive_metrics.get("observed_modality_max")
                    ),
                }
            )
        if self.contrastive_diagnostics_frequency:
            n_low, n_high = _feature_low_high_ratios(self.filter, features.numerical)
            s_low, s_high = _feature_low_high_ratios(self.filter, features.structural)
            row.update(
                {
                    "numerical_low_energy_ratio": float(n_low.mean().cpu()),
                    "numerical_high_energy_ratio": float(n_high.mean().cpu()),
                    "structural_low_energy_ratio": float(s_low.mean().cpu()),
                    "structural_high_energy_ratio": float(s_high.mean().cpu()),
                }
            )
        if self.contrastive_diagnostics_reliability:
            for modality in STANDARD_MODALITIES:
                row[f"reliability_{modality}_structural"] = _metric_float(
                    self._last_contrastive_metrics.get(f"reliability_{modality}_structural")
                )
                row[f"reliability_{modality}_numerical"] = _metric_float(
                    self._last_contrastive_metrics.get(f"reliability_{modality}_numerical")
                )
        return row

    def _write_diagnostics(
        self,
        prefix: str,
        batch: BGSampleBatch,
        features,
        bg,
        gate,
        rho_true,
        loss,
        stage_metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self.diagnostics_enabled or not self._is_global_zero():
            return
        key = f"{prefix}-{self.stage}"
        if key in self._diagnostic_written:
            return
        self._diagnostic_written.add(key)
        output_dir = self.diagnostics_output_dir / self.stage / prefix
        if self.stage == "contrastive":
            save_tensor_panel(
                {
                    "N": features.numerical[:, :1],
                    "S": features.structural[:, :1],
                    "U": features.unique[:, :1],
                },
                output_dir / "feature_panel.png",
                cmap="viridis",
            )
            append_metrics_csv(
                self.diagnostics_output_dir / "metrics.csv",
                {
                    "stage": self.stage,
                    "split": prefix,
                    "epoch": int(getattr(self, "current_epoch", 0)),
                    "loss": float(loss.detach().cpu()),
                    "stage_checkpoint_used": self.stage_checkpoint_used,
                    "encoder_frozen": str(not any(parameter.requires_grad for parameter in self.encoder.parameters())),
                    **self._contrastive_metric_row(features),
                },
            )
            return
        assert bg is not None and gate is not None and rho_true is not None
        low, _ = self._anchors(batch.depth_vel.detach())
        bg_high = self.filter.highpass(bg.bg_hat.detach())
        save_tensor_panel(
            {
                "depth_vel": batch.depth_vel,
                "P_L(V)": low,
                "B_hat": bg.bg_hat,
                "P_H(B_hat)": bg_high.abs(),
            },
            output_dir / "background_panel.png",
        )
        save_tensor_panel(
            {
                "N": features.numerical[:, :1],
                "S": features.structural[:, :1],
                "U": features.unique[:, :1],
            },
            output_dir / "feature_panel.png",
            cmap="viridis",
        )
        save_metric_pair_plot(rho_true, gate.rho_hat_b, output_dir / "rho_calibration.png")
        metric_row = {
            "stage": self.stage,
            "split": prefix,
            "epoch": int(getattr(self, "current_epoch", 0)),
            "loss": float(loss.detach().cpu()),
            "epsilon_H_B": float(bg.epsilon_h_b.mean().detach().cpu()),
            "rho_B": float(rho_true.mean().detach().cpu()),
            "rho_hat_B": float(gate.rho_hat_b.mean().detach().cpu()),
            "alpha_hat": float(gate.alpha_hat.mean().detach().cpu()),
            "stage_checkpoint_used": self.stage_checkpoint_used,
            "encoder_frozen": str(not any(parameter.requires_grad for parameter in self.encoder.parameters())),
            "codec_type": self.codec_type,
            "residual_backend": self.residual_backend,
            "residual_num_inference_steps": self.residual_num_inference_steps,
            "symile_structural": _metric_float(self._last_contrastive_metrics.get("symile_structural")),
            "symile_numerical": _metric_float(self._last_contrastive_metrics.get("symile_numerical")),
            "anchor": _metric_float(self._last_contrastive_metrics.get("anchor")),
            "reliability_prior": _metric_float(self._last_contrastive_metrics.get("reliability_prior")),
            "sn_ortho": _metric_float(self._last_contrastive_metrics.get("sn_ortho")),
            "unique_ortho": _metric_float(self._last_contrastive_metrics.get("unique_ortho")),
        }
        for name, value in (stage_metrics or {}).items():
            metric_row[name] = _metric_float(value)
        append_metrics_csv(
            self.diagnostics_output_dir / "metrics.csv",
            metric_row,
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    @torch.no_grad()
    def predict_batch(
        self,
        batch: BGInferenceBatch | BGSampleBatch | Mapping[str, torch.Tensor],
    ) -> PredictionBatch:
        batch = self._as_inference_batch(batch)
        if self.stage == "contrastive":
            raise RuntimeError("Contrastive-only stage does not support velocity prediction.")
        residual_override = str(_conf_get(self.conf, "evaluation.residual_override", "none")).lower()
        if self.stage != "background":
            residual_background_source = str(
                _conf_get(self.conf, "model.residual_background_source", "predicted")
            ).lower()
            if residual_background_source != "predicted":
                raise ValueError(
                    "model.residual_background_source must be 'predicted' for observed-only residual inference; "
                    f"got {residual_background_source!r}."
                )
            if residual_override in {"oracle", "oracle_latent"}:
                raise ValueError(
                    "evaluation.residual_override values 'oracle' and 'oracle_latent' require target-derived "
                    "residuals and are not available during observed-only inference."
                )
        _, conditions, bg, gate = self._common_forward(batch, use_background_target=False)
        assert bg is not None and gate is not None
        assert self.residual is not None and self.codec is not None
        if self.stage == "background":
            velocity_hat = bg.bg_hat
            residual_hat = torch.zeros_like(velocity_hat)
        else:
            assert conditions is not None
            residual_bg_hat = bg.bg_hat.detach()
            if residual_override in {"zero", "zero_raw", "background"}:
                velocity_hat = residual_bg_hat
            else:
                z_bg_hat = self.codec.encode(residual_bg_hat, deterministic=True)
                if residual_override in {"decode_bg", "decode_bg_only", "zero_latent"}:
                    z_res_hat = torch.zeros_like(z_bg_hat)
                elif residual_override in {"none", ""}:
                    cond_s = conditions.structural
                    structural_ablation = str(_conf_get(self.conf, "evaluation.structural_ablation", "none")).lower()
                    if structural_ablation == "zero":
                        cond_s = torch.zeros_like(cond_s)
                    elif structural_ablation == "shuffle":
                        cond_s = cond_s.roll(shifts=1, dims=0)
                    elif structural_ablation not in {"none", ""}:
                        raise ValueError(
                            "evaluation.structural_ablation must be one of 'none', 'zero', or 'shuffle', "
                            f"got {structural_ablation!r}."
                        )
                    z_res_hat = self.residual.sample(
                        cond_s,
                        residual_bg_hat.detach(),
                        x_size=tuple(z_bg_hat.shape),
                        steps=self.residual_num_inference_steps,
                        z_bg=z_bg_hat,
                    )
                else:
                    raise ValueError(
                        "evaluation.residual_override must be one of 'none', 'zero', 'decode_bg', "
                        f"or 'zero_latent', got {residual_override!r}."
                    )
                velocity_hat = self.codec.decode(z_bg_hat + z_res_hat, out_hw=batch.output_hw)
            residual_hat = velocity_hat - residual_bg_hat
        return PredictionBatch(
            bg_hat=residual_bg_hat if self.stage != "background" else bg.bg_hat,
            velocity_hat=velocity_hat,
            residual_hat=residual_hat,
            rho_hat_b=gate.rho_hat_b,
            alpha_hat=gate.alpha_hat,
            epsilon_h_b=bg.epsilon_h_b,
            metadata=batch.metadata,
        )

    def configure_optimizers(self):
        if self.stage == "joint_full":
            def trainable_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
                return [parameter for parameter in module.parameters() if parameter.requires_grad]

            groups: list[dict[str, object]] = []
            encoder_backbone_params: list[torch.nn.Parameter] = []
            encoder_head_params: list[torch.nn.Parameter] = []
            for name, parameter in self.encoder.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".heads." in name or name.startswith("heads.") or ".head" in name or "head." in name:
                    encoder_head_params.append(parameter)
                else:
                    encoder_backbone_params.append(parameter)

            if encoder_backbone_params:
                groups.append({
                    "params": encoder_backbone_params,
                    "lr": float(_conf_get(self.conf, "training.joint.encoder_backbone_lr", 1e-6)),
                })
            if encoder_head_params:
                groups.append({
                    "params": encoder_head_params,
                    "lr": float(_conf_get(self.conf, "training.joint.encoder_head_lr", 3e-6)),
                })
            contrastive_params = trainable_params(self.contrastive_loss)
            if contrastive_params:
                groups.append({
                    "params": contrastive_params,
                    "lr": float(_conf_get(self.conf, "training.joint.contrastive_lr", 3e-6)),
                })
            assert self.background is not None and self.residual is not None and self.adapters is not None
            background_params = trainable_params(self.background)
            if self.background_raw_bypass is not None:
                background_params.extend(trainable_params(self.background_raw_bypass))
            if background_params:
                groups.append({
                    "params": background_params,
                    "lr": float(_conf_get(self.conf, "training.joint.background_lr", 1e-4)),
                })
            residual_params = trainable_params(self.residual)
            if residual_params:
                groups.append({
                    "params": residual_params,
                    "lr": float(_conf_get(self.conf, "training.joint.residual_lr", 5e-5)),
                })
            adapter_params = trainable_params(self.adapters)
            if adapter_params:
                groups.append({
                    "params": adapter_params,
                    "lr": float(_conf_get(self.conf, "training.joint.adapter_lr", 3e-6)),
                })
            if not groups:
                raise RuntimeError("No trainable parameter groups for joint_full stage.")
            return torch.optim.AdamW(
                groups,
                weight_decay=float(_conf_get(self.conf, "training.weight_decay", 0.0)),
            )
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable parameters for BG-PDR-FM stage {self.stage!r}.")
        return torch.optim.AdamW(trainable, lr=self.lr)

    def on_train_epoch_end(self) -> None:
        self._save_stage_checkpoint()

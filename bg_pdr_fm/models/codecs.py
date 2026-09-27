"""Latent codec abstractions used by background and residual generators."""

from __future__ import annotations

from pathlib import Path
import importlib
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCodec(Protocol):
    """Minimal latent interface required by BG-PDR-FM generators."""

    def encode(self, image: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        ...

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        ...


class SimpleLatentCodec(nn.Module):
    """Small trainable codec for smoke runs and checkpoint-free validation."""

    codec_type = "simple"
    trainable_codec = True

    def __init__(self, image_channels: int = 1, latent_channels: int = 16, latent_hw: tuple[int, int] = (16, 16)) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_hw = tuple(latent_hw)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1),
        )

    def encode(self, image: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        x = F.interpolate(image, size=self.latent_hw, mode="bilinear", align_corners=False)
        return self.encoder(x)

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int] = (70, 70)) -> torch.Tensor:
        x = self.decoder(latent)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class IdentityLatentCodec(nn.Module):
    """Parameter-free codec that runs residual generation directly in image space."""

    codec_type = "identity"
    trainable_codec = False

    def __init__(self, image_channels: int = 1, latent_channels: int = 1, latent_hw: tuple[int, int] = (70, 70)) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.latent_channels = int(latent_channels)
        self.latent_hw = tuple(int(v) for v in latent_hw)
        if self.image_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("IdentityLatentCodec channel counts must be positive.")
        if self.image_channels != self.latent_channels:
            raise ValueError(
                "IdentityLatentCodec requires image_channels == latent_channels; "
                f"got image_channels={self.image_channels}, latent_channels={self.latent_channels}."
            )
        if len(self.latent_hw) != 2 or min(self.latent_hw) <= 0:
            raise ValueError(f"IdentityLatentCodec latent_hw must be a positive HW pair, got {latent_hw!r}.")

    def _validate(self, tensor: torch.Tensor, label: str) -> None:
        expected = (self.latent_channels, *self.latent_hw)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected:
            raise ValueError(
                f"IdentityLatentCodec {label} expected BCHW with trailing shape {expected}, "
                f"got {tuple(tensor.shape)}."
            )

    def encode(self, image: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        self._validate(image, "encode")
        return image

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        self._validate(latent, "decode")
        if tuple(int(v) for v in out_hw) != self.latent_hw:
            raise ValueError(
                f"IdentityLatentCodec decode out_hw must match latent_hw={self.latent_hw}, got {out_hw!r}."
            )
        return latent


class AutoencoderLatentCodec(nn.Module):
    """Checkpoint-backed adapter around an AutoencoderKLLightning VAE."""

    codec_type = "autoencoder"
    trainable_codec = False

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        module_path: str | None = None,
        loader_cls: type[Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            raise FileNotFoundError(
                "AutoencoderLatentCodec requires `model.codec_checkpoint`. "
                "Set a valid checkpoint path or use `model.codec_type: simple` for smoke runs."
            )
        self.checkpoint_path = Path(checkpoint_path)
        self.module_path = module_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AutoencoderLatentCodec checkpoint not found: {self.checkpoint_path}. "
                "This backend does not silently fall back to SimpleLatentCodec."
            )
        if loader_cls is None:
            loader_cls = self._import_legacy_loader(module_path)
        try:
            autoencoder = loader_cls.load_from_checkpoint(str(self.checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load AutoencoderKLLightning checkpoint for AutoencoderLatentCodec: "
                f"{self.checkpoint_path}"
            ) from exc
        if not hasattr(autoencoder, "vae"):
            raise AttributeError(
                "AutoencoderLatentCodec expected the loaded checkpoint module to expose `.vae`, "
                f"got {type(autoencoder).__name__}."
            )
        self.vae = autoencoder.vae
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad = False

    @classmethod
    def _import_legacy_loader(cls, module_path: str | None = None) -> type[Any]:
        target = module_path or "bg_pdr_fm.lightning.autoencoder_module:AutoencoderKLLightning"
        if ":" in target:
            module_name, class_name = target.split(":", 1)
        else:
            module_name, class_name = target.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(
                f"Failed to import autoencoder loader {target!r}. "
                "Use a standalone bg_pdr_fm loader module."
            ) from exc
        if not hasattr(module, class_name):
            raise AttributeError(f"Autoencoder loader module {module_name!r} has no class {class_name!r}.")
        return getattr(module, class_name)

    @staticmethod
    def _latent_from_posterior(posterior: Any, deterministic: bool) -> torch.Tensor:
        if torch.is_tensor(posterior):
            return posterior
        if hasattr(posterior, "latent_dist"):
            return AutoencoderLatentCodec._latent_from_posterior(posterior.latent_dist, deterministic)
        method_name = "mode" if deterministic else "sample"
        if hasattr(posterior, method_name):
            latent = getattr(posterior, method_name)()
            if torch.is_tensor(latent):
                return latent
        raise TypeError(
            "AutoencoderLatentCodec encode expected a tensor, a posterior with `.mode()`/`.sample()`, "
            "or an object with `.latent_dist`."
        )

    def encode(self, image: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.vae.encode(image)
            latent = self._latent_from_posterior(posterior, deterministic=deterministic)
        if latent.ndim != 4:
            raise ValueError(f"AutoencoderLatentCodec encode must return BCHW latent, got {tuple(latent.shape)}.")
        return latent

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        reconstruction = self.vae.decode(latent)
        if not torch.is_tensor(reconstruction):
            if hasattr(reconstruction, "sample") and torch.is_tensor(reconstruction.sample):
                reconstruction = reconstruction.sample
            else:
                raise TypeError("AutoencoderLatentCodec decode expected a tensor-like reconstruction.")
        if reconstruction.ndim != 4:
            raise ValueError(
                f"AutoencoderLatentCodec decode must return BCHW image, got {tuple(reconstruction.shape)}."
            )
        if tuple(reconstruction.shape[-2:]) != tuple(out_hw):
            reconstruction = F.interpolate(reconstruction, size=out_hw, mode="bilinear", align_corners=False)
        return reconstruction


def build_latent_codec(
    codec_type: str = "simple",
    latent_channels: int = 16,
    latent_hw: tuple[int, int] = (16, 16),
    checkpoint_path: str | Path | None = None,
    module_path: str | None = None,
) -> nn.Module:
    codec_type = str(codec_type).lower()
    if codec_type == "simple":
        return SimpleLatentCodec(latent_channels=latent_channels, latent_hw=latent_hw)
    if codec_type in {"identity", "pixel", "none"}:
        return IdentityLatentCodec(latent_channels=latent_channels, latent_hw=latent_hw)
    if codec_type in {"autoencoder", "vae"}:
        return AutoencoderLatentCodec(checkpoint_path=checkpoint_path, module_path=module_path)
    raise ValueError(
        f"Unknown model.codec_type: {codec_type!r}. Expected 'simple', 'identity', or 'autoencoder'."
    )

"""Lightning module for standalone seismic autoencoder training."""

from __future__ import annotations

import math
from typing import Any

import lightning
import torch
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf

from bg_pdr_fm.models.autoencoder import SeismicAutoencoderKL


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    value = OmegaConf.select(conf, path, default=default)
    return value


class SeismicAutoencoderKLLightning(lightning.LightningModule):
    """Lightning module for BG-PDR-FM's seismic KL autoencoder."""

    def __init__(self, conf: Any) -> None:
        super().__init__()
        conf = OmegaConf.create(conf)
        self.save_hyperparameters({"conf": OmegaConf.to_container(conf, resolve=False)})
        self.conf = conf
        self.lr = float(_conf_get(conf, "training.lr", 1e-4))
        self.batch_size = int(_conf_get(conf, "training.dataloader.batch_size", 64))
        ae_conf = _conf_get(conf, "autoencoder_conf", {})
        self.vae = SeismicAutoencoderKL(
            latent_channels=int(_conf_get(ae_conf, "latent_channels", 4)),
            depth_vel_shape=tuple(_conf_get(conf, "datasets.depth_velocity.shape", [1, 70, 70])),
            down_block_types=tuple(_conf_get(ae_conf, "down_block_types", ("DownEncoderBlock2D",) * 3)),
            up_block_types=tuple(_conf_get(ae_conf, "up_block_types", ("UpDecoderBlock2D",) * 3)),
            block_out_channels=tuple(_conf_get(ae_conf, "block_out_channels", (64, 128, 256))),
            sample_size=int(_conf_get(ae_conf, "sample_size", 72)),
            layers_per_block=int(_conf_get(ae_conf, "layers_per_block", 2)),
            norm_num_groups=int(_conf_get(ae_conf, "norm_num_groups", 32)),
        )
        self._last_train_batch = None
        self._last_val_batch = None
        self.ema = None
        self._ema_parameters = None
        self.ka = {
            "strategy": str(_conf_get(conf, "training.kl_anneal.strategy", "linear_epoch")),
            "warmup_epochs": int(_conf_get(conf, "training.kl_anneal.warmup_epochs", 30)),
            "start": float(_conf_get(conf, "training.kl_anneal.start", 0.0)),
            "end": float(_conf_get(conf, "training.kl_anneal.end", _conf_get(conf, "training.loss.kl_weight", 5e-4))),
            "cycles": int(_conf_get(conf, "training.kl_anneal.cycles", 1)),
            "ratio": float(_conf_get(conf, "training.kl_anneal.ratio", 0.5)),
            "free_bits": float(_conf_get(conf, "training.kl_anneal.free_bits", 0.0)),
        }
        self.register_buffer("kl_beta", torch.tensor(float(self.ka["start"])))
        if bool(_conf_get(conf, "training.use_ema", True)):
            self._ema_parameters = list(self.vae.parameters())
            self.ema = EMAModel(
                parameters=self._ema_parameters,
                use_ema_warmup=True,
                foreach=True,
                power=0.75,
                device=self.device,
            )

    def _ema_params(self):
        params = self._ema_parameters if self._ema_parameters is not None else list(self.parameters())
        if self.ema is not None:
            try:
                param_device = next(p.device for p in params if p.requires_grad)
            except StopIteration:
                param_device = None
            if param_device is not None and len(self.ema.shadow_params) > 0:
                if self.ema.shadow_params[0].device != param_device:
                    self.ema.to(param_device)
        return params

    def setup(self, stage: str | None = None) -> None:
        if self.ema is not None:
            self.ema.to(self.device)

    def on_fit_start(self) -> None:
        if self.ema is not None:
            self.ema.to(self.device)

    def on_train_epoch_start(self) -> None:
        self.kl_beta.fill_(self._compute_beta_epoch(self.current_epoch))

    def _compute_beta_epoch(self, epoch: int) -> float:
        strategy = str(self.ka["strategy"])
        start, end = float(self.ka["start"]), float(self.ka["end"])
        warmup = max(1, int(self.ka["warmup_epochs"]))
        if strategy == "none":
            return end
        if strategy == "linear_epoch":
            t = min(1.0, (epoch + 1) / warmup)
            return start + t * (end - start)
        if strategy == "cosine_epoch":
            t = 0.5 * (1.0 - math.cos(math.pi * min(epoch + 1, warmup) / warmup))
            return start + t * (end - start)
        if strategy == "cyclic_epoch":
            cycles = max(1, int(self.ka["cycles"]))
            total = max(1, int(getattr(self.trainer, "max_epochs", warmup)))
            cycle_len = max(1, total // cycles)
            rise_len = max(1, int(cycle_len * float(self.ka["ratio"])))
            pos = (epoch % cycle_len) + 1
            t = min(1.0, pos / rise_len)
            return start + t * (end - start)
        return end

    @staticmethod
    def finite_difference_loss(pred: torch.Tensor, target: torch.Tensor, w_y: float = 1.0, w_x: float = 0.5):
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        return w_y * F.l1_loss(dy_pred, dy_target) + w_x * F.l1_loss(dx_pred, dx_target)

    @staticmethod
    def high_frequency_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 5):
        padding = kernel_size // 2
        pred_low = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=padding)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
        return F.l1_loss(pred - pred_low, target - target_low)

    def _kl_loss(self, posterior) -> tuple[torch.Tensor, torch.Tensor]:
        kl_map = posterior.kl()
        kl_raw = kl_map.mean()
        free_bits = float(self.ka["free_bits"])
        if free_bits <= 0.0:
            return kl_raw, kl_raw
        batch_size = kl_map.shape[0]
        kl_per_sample = kl_map.view(batch_size, -1).sum(dim=1)
        kl_used = torch.clamp(kl_per_sample - free_bits, min=0.0).mean()
        return kl_raw, kl_used

    def _shared_step(self, batch, prefix: str):
        depth_velocity = batch["depth_vel"]
        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents)

        l1_loss = F.l1_loss(reconstructions, depth_velocity)
        mse_loss = F.mse_loss(reconstructions, depth_velocity)
        grad_loss = self.finite_difference_loss(reconstructions, depth_velocity)
        hf_loss = self.high_frequency_loss(reconstructions, depth_velocity)
        kl_loss_raw, kl_loss_used = self._kl_loss(posterior)

        beta = float(self.kl_beta.item())
        loss = (
            l1_loss * float(_conf_get(self.conf, "training.loss.l1_weight", 1.0))
            + mse_loss * float(_conf_get(self.conf, "training.loss.mse_weight", 0.1))
            + grad_loss * float(_conf_get(self.conf, "training.loss.gradient_weight", 0.5))
            + hf_loss * float(_conf_get(self.conf, "training.loss.high_frequency_weight", 0.1))
            + kl_loss_used * beta
        )

        self.log(f"{prefix}/loss", loss.detach(), on_step=(prefix == "train"), on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)
        self.log_dict(
            {
                f"{prefix}/MAE": l1_loss.detach(),
                f"{prefix}/MSE": mse_loss.detach(),
                f"{prefix}/grad": grad_loss.detach(),
                f"{prefix}/high_freq": hf_loss.detach(),
                f"{prefix}/KL_raw": kl_loss_raw.detach(),
                f"{prefix}/KL_used": kl_loss_used.detach(),
                f"{prefix}/beta": torch.tensor(beta, device=self.device),
                f"{prefix}/latent_std": latents.detach().std(),
            },
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        return loss, depth_velocity, reconstructions

    def training_step(self, batch, batch_idx):
        loss, depth_velocity, reconstructions = self._shared_step(batch, "train")
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())
        if self.ema is not None:
            self.ema.step(self._ema_params())
        return loss

    def on_validation_epoch_start(self) -> None:
        if self.ema is not None:
            params = self._ema_params()
            self.ema.store(params)
            self.ema.copy_to(params)

    def validation_step(self, batch, batch_idx):
        loss, depth_velocity, reconstructions = self._shared_step(batch, "val")
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.ema is not None:
            self.ema.restore(self._ema_params())

    def test_step(self, batch, batch_idx):
        loss, depth_velocity, reconstructions = self._shared_step(batch, "test")
        return loss

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema" not in checkpoint:
            return
        if self.ema is None:
            self.ema = EMAModel(parameters=self._ema_params())
        self.ema.load_state_dict(checkpoint["ema"])

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)


AutoencoderKLLightning = SeismicAutoencoderKLLightning

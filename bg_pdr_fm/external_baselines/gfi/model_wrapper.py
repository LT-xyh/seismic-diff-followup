"""Shape-checking wrapper for externally supplied GFI models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


EXPECTED_INPUT_SHAPE = (5, 1000, 70)
EXPECTED_OUTPUT_SHAPE = (1, 70, 70)


class GFIModelWrapper(nn.Module):
    """Wrap an external GFI predictor with explicit tensor contract checks."""

    def __init__(
        self,
        predictor: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        *,
        expected_input_shape: tuple[int, ...] = EXPECTED_INPUT_SHAPE,
        expected_output_shape: tuple[int, ...] = EXPECTED_OUTPUT_SHAPE,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.expected_input_shape = tuple(int(dim) for dim in expected_input_shape)
        self.expected_output_shape = tuple(int(dim) for dim in expected_output_shape)

    def _check_input(self, waveform: torch.Tensor) -> None:
        if waveform.dim() != 4:
            raise ValueError(f"GFI input must have rank 4, got shape {tuple(waveform.shape)}")
        if tuple(int(dim) for dim in waveform.shape[1:]) != self.expected_input_shape:
            raise ValueError(
                f"GFI input must have shape [B, {self.expected_input_shape}], got {tuple(waveform.shape)}"
            )

    def _check_output(self, velocity: torch.Tensor) -> None:
        if velocity.dim() != 4:
            raise ValueError(f"GFI output must have rank 4, got shape {tuple(velocity.shape)}")
        if tuple(int(dim) for dim in velocity.shape[1:]) != self.expected_output_shape:
            raise ValueError(
                f"GFI output must have shape [B, {self.expected_output_shape}], got {tuple(velocity.shape)}"
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self._check_input(waveform)
        velocity = self.predictor(waveform)
        if not isinstance(velocity, torch.Tensor):
            raise TypeError(f"GFI predictor must return a torch.Tensor, got {type(velocity)!r}")
        self._check_output(velocity)
        return velocity

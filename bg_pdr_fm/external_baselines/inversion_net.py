"""Official-capacity InversionNet adapted to the multimodal input protocol."""

from __future__ import annotations

import torch
import torch.nn as nn

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image
from bg_pdr_fm.external_baselines.openfwi_official import OpenFWIInversionNet


class AdaptedInversionNetMultimodal(nn.Module):
    def __init__(self, input_hw: tuple[int, int] = (1000, 70)) -> None:
        super().__init__()
        self.input_hw = tuple(int(value) for value in input_hw)
        if self.input_hw != (1000, 70):
            raise ValueError(f"Official InversionNet requires input_hw=(1000, 70), got {self.input_hw}.")
        self.model = OpenFWIInversionNet()

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        return self.model(build_multimodal_condition_image(batch, self.input_hw))

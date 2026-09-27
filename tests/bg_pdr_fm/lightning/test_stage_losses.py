import unittest

import torch

from bg_pdr_fm.lightning.stage_losses import select_residual_background
from bg_pdr_fm.models.filters import LowHighPassFilter


class ResidualBackgroundSourceTest(unittest.TestCase):
    def test_predicted_background_source_is_default(self):
        target = torch.randn(2, 1, 8, 8)
        predicted = torch.randn(2, 1, 8, 8)

        selected = select_residual_background({}, LowHighPassFilter(kernel_size=5), target, predicted)

        self.assertTrue(torch.equal(selected, predicted))

    def test_oracle_lowpass_background_source_uses_lowpass_target(self):
        target = torch.randn(2, 1, 8, 8)
        predicted = torch.randn(2, 1, 8, 8)
        filter_module = LowHighPassFilter(kernel_size=5)

        selected = select_residual_background(
            {"model": {"residual_background_source": "oracle_lowpass"}},
            filter_module,
            target,
            predicted,
        )

        self.assertTrue(torch.equal(selected, filter_module.lowpass(target)))


if __name__ == "__main__":
    unittest.main()

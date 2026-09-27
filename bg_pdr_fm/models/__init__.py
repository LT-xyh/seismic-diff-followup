"""Public model exports for the self-contained BG-PDR-FM package."""

from .adapters import (
    ConditionAdapters,
    NumericalConditionAdapter,
    StructuralConditionAdapter,
    UniqueConditionAdapter,
)
from .autoencoder import SeismicAutoencoderKL
from .calibrator import ResidualGate, RhoCalibrator
from .codecs import AutoencoderLatentCodec, IdentityLatentCodec, LatentCodec, SimpleLatentCodec, build_latent_codec
from .contrastive import (
    DepthVelocityWaveletTransform,
    PhysicsAnchoredSubsetSymileLoss,
    build_contrastive_loss,
    build_wavelet_anchor,
)
from .encoder import PhysicsDecoupledEncoder
from .filters import LowHighPassFilter
from .generators import BackgroundEstimator, ResidualFlowGenerator
from .modality_encoders import HorizonEncoderA, RMSVelocityEncoderA, SeismicImageEncoderA, WellLogEncoderA
from .residual_backends import (
    DiffusersCrossAttnResidualFMBackend,
    LegacyLatentFlowMatchingBackend,
    SimpleLatentDDPMBackend,
    SimpleResidualFMBackend,
    build_residual_backend,
)
from .types import BackgroundEstimate, Conditions, DecoupledFeatures, PredictionBatch

__all__ = [
    "BackgroundEstimate",
    "BackgroundEstimator",
    "AutoencoderLatentCodec",
    "IdentityLatentCodec",
    "ConditionAdapters",
    "Conditions",
    "DecoupledFeatures",
    "DepthVelocityWaveletTransform",
    "DiffusersCrossAttnResidualFMBackend",
    "HorizonEncoderA",
    "LatentCodec",
    "LegacyLatentFlowMatchingBackend",
    "LowHighPassFilter",
    "NumericalConditionAdapter",
    "PhysicsDecoupledEncoder",
    "PhysicsAnchoredSubsetSymileLoss",
    "PredictionBatch",
    "RMSVelocityEncoderA",
    "ResidualFlowGenerator",
    "ResidualGate",
    "RhoCalibrator",
    "SeismicImageEncoderA",
    "SeismicAutoencoderKL",
    "build_residual_backend",
    "SimpleResidualFMBackend",
    "SimpleLatentDDPMBackend",
    "SimpleLatentCodec",
    "StructuralConditionAdapter",
    "UniqueConditionAdapter",
    "WellLogEncoderA",
    "build_contrastive_loss",
    "build_latent_codec",
    "build_residual_backend",
    "build_wavelet_anchor",
]

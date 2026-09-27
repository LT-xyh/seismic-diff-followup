# Training-free motivation statistics

This report was generated from normalized OpenFWI observations and targets without constructing or loading a neural network, checkpoint, optimizer, or learned dimensionality-reduction method.

## Protocol

- Global split: 70/20/10, split seed `42`; only the global training split was read.
- Analysis seed: `2027`; well seed: `1234` with deterministic well generation.
- Subsets: FlatVelA, FlatVelB, CurveVelA, CurveVelB, FlatFaultA, FlatFaultB, CurveFaultA, CurveFaultB.
- Samples per subset requested: `1000`; selected counts: `{'FlatVelA': 1000, 'FlatVelB': 1000, 'CurveVelA': 1000, 'CurveVelB': 1000, 'FlatFaultA': 1000, 'FlatFaultB': 1000, 'CurveFaultA': 1000, 'CurveFaultB': 1000}`.
- Low-pass operator: stride-one `5x5` average pooling with zero padding and padding included in the divisor.
- Maximum numerical error in `V = B + S`: `1.110e-16`.

## Figure 1a: data-level kernel alignment

The entries below are median RBF CKA values across the eight subsets. They are reported as data-level kernel-alignment or dependence proxies, not mutual information, causality, or identifiable semantic roles.

| Modality | Background B | Structure S |
|---|---:|---:|
| PoSTM | 0.3083 | 0.3000 |
| Horizon | 0.1514 | 0.3676 |
| RMS velocity | 0.8201 | 0.4954 |
| Well log + mask | 0.0395 | 0.0942 |

Matched versus fixed-shuffle controls:

| Modality | Component | Matched median | Shuffled median | Difference |
|---|---|---:|---:|---:|
| Horizon | Background B | 0.1514 | 0.0453 | 0.1061 |
| Horizon | Structure S | 0.3676 | 0.1473 | 0.2203 |
| PoSTM | Background B | 0.3083 | 0.0129 | 0.2954 |
| PoSTM | Structure S | 0.3000 | 0.0390 | 0.2610 |
| RMS velocity | Background B | 0.8201 | 0.0072 | 0.8129 |
| RMS velocity | Structure S | 0.4954 | 0.0181 | 0.4774 |
| Well log + mask | Background B | 0.0395 | 0.0318 | 0.0077 |
| Well log + mask | Structure S | 0.0942 | 0.0838 | 0.0104 |

## Figure 1b: effective dimension

The effective rank is the participation-ratio summary of the nonzero sample-covariance spectrum. It measures intrinsic variability/effective degrees of freedom and is not conditional entropy.

| Component | Median $r_{\mathrm{eff}}$ | Mean | Std | IQR |
|---|---:|---:|---:|---:|
| Background B | 4.954 | 5.399 | 2.777 | 4.590 |
| Structure S | 54.030 | 138.540 | 135.369 | 183.878 |

### Per-subset effective ranks

| Subset | Background B | Structure S | Structure minus background |
|---|---:|---:|---:|
| FlatVel-A | 2.024 | 24.261 | 22.237 |
| FlatVel-B | 5.663 | 39.672 | 34.008 |
| CurveVel-A | 2.588 | 58.522 | 55.934 |
| CurveVel-B | 7.530 | 200.888 | 193.358 |
| FlatFault-A | 3.061 | 37.829 | 34.768 |
| FlatFault-B | 7.544 | 289.694 | 282.150 |
| CurveFault-A | 4.245 | 49.539 | 45.294 |
| CurveFault-B | 10.536 | 407.912 | 397.377 |

## Interpretation

- Structure effective rank exceeds background effective rank in `8/8` subsets; median difference is `50.614`.
- Across all modality/subset pairs, structure-minus-background alignment is positive in `20/32` cases.
- These results support or weaken the proposed information-source and degree-of-freedom asymmetries only at the level of the stated data statistics. They do not prove representation identifiability, causal modality responsibility, or conditional-entropy ordering.

## Limitations

- The analysis uses a fixed training-only sample and fixed median-heuristic bandwidths; no sample, axis, or bandwidth was adjusted after seeing the result.
- Well logs and masks are concatenated as one raw information block; the mask is not discarded.
- The effective-dimension calculation intentionally avoids per-pixel whitening and per-sample unit-norm scaling, so its values describe the normalized fields under the specified protocol.

## Provenance

- Per-subset metadata and shuffle permutations are recorded in `analysis_config.json`; the complete selected-record manifest is `analysis_manifest.csv`.

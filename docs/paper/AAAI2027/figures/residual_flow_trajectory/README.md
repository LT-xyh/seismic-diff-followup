# Prediction-Consistent Residual Flow Trajectory

This directory contains a real trajectory captured from the formal PD-BG-RFM
checkpoint. No proxy trajectory, interpolation-only state, spatial warp, or
post-hoc RFW registration was used.

## Provenance

- Checkpoint: /public/home/xuyinghao/workspace/seismic-diff/logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/lightning/checkpoints/joint_full_contrastive_bgfm_pixelfm_predbg_e100-epoch_99-loss0.1036.ckpt
- Checkpoint SHA256: 0d77a529618d8d158af0731752cc68efc01c4f6e2f2adfedaf87d8008f930a36
- Config: /public/home/xuyinghao/workspace/seismic-diff/bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml
- Official held-out manifest: /public/home/xuyinghao/workspace/seismic-diff/logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/evaluation_epoch99_global_heldout_full_parallel_bs64/held_out_manifest.csv
- Official metrics: /public/home/xuyinghao/workspace/seismic-diff/logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/evaluation_epoch99_global_heldout_full_parallel_bs64/merged_metrics.csv
- Selected subset: FlatFaultB
- Dataset split used to load the asset: all
- Provenance scope: not_in_official_global_heldout
- Selection rule: user-specified Fig. 2 asset: FlatFaultB source_sample_index=27006; not in official global held-out manifest
- Selected dataset index: 28286
- Selected source sample index: 27006
- Record ID: dataset_id=5, dataset_name=FlatFaultB, source_sample_index=27006
- Official metrics: not reported; this record is outside the official global held-out manifest.
- Evaluation row: none (visualization asset only).
- Trace seed: 2027
- Euler steps: 50
- CUDA_VISIBLE_DEVICES: 3
- Color limits:
  - feature RMS: [0.0, 1.2106242501735605]
  - residual state: [-3.8077375888824463, 3.8077375888824463]
  - predicted field: [-3.8656580448150635, 3.8656580448150635]
  - background and final velocity: [-0.9458022713661194, 0.9458022713661194]
  - final residual: [-0.5660951733589172, 0.5660951733589172]

## Exact sampler

The formal model uses identity/pixel-space residual Flow Matching:

    c_R = [C_str; psi_bg(B_hat)]
    R_0 = xi, xi ~ N(0, I)
    t_k = k / 50
    v_k = v_theta(R_k, t_k, c_R)
    R_(k+1) = R_k + (1/50) v_k
    V_hat = B_hat + R_hat

trajectory_arrays.npz stores the real R_k, v_k, Delta R_k, time values,
conditions, background, residual, and final velocity. The stored state arrays
have per-state shape 1 x 70 x 70.

## Validation

- Exact checkpoint load: missing=0, unexpected=0.
- Traced R_50 vs production sample() under the same noise, dtype, and
  condition: torch.testing.assert_close passed.
- Composition V_hat = B_hat + R_hat: torch.testing.assert_close passed.
- All saved arrays are finite.

See trajectory_metadata.json for complete shapes, color limits, hashes, and
validation values. The PPTX keeps labels, equations, and arrows editable; the
spatial tiles are embedded raster data assets generated directly from tensors.

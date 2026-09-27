# AAAI27 Benchmark Configs

Official benchmark configs for the paper comparison pipeline live here. The legacy
local `src/` training tree has been removed from the active project and must not
be used for AAAI27 tables.

Run smoke training:

```bash
python -m bg_pdr_fm.training.train_aaai27_benchmark \
  --config bg_pdr_fm/configs/experiments/aaai27/concat_fm_strong.yaml \
  --fast-run
```

Run unified smoke evaluation without a checkpoint:

```bash
python -m bg_pdr_fm.evaluation.compare_experiments \
  --config bg_pdr_fm/configs/experiments/aaai27/concat_fm_strong.yaml
```

Main missing-modality modes are fixed to `full`, `w/o well_log`, `w/o horizon`,
`w/o rms_vel`, `w/o well+rms`, and `PSTM only`.

Run formal benchmark training on idle GPUs:

```bash
/public/home/xuyinghao/miniconda3/envs/seg/bin/python \
  -m bg_pdr_fm.training.launch_aaai27_formal \
  --gpus 0 1 2 3
```

Formal configs are named `formal_*.yaml` and use OpenFWI LMDB, the frozen
contrastive encoder checkpoint, the same `LowHighPassFilter`, and per-method
checkpoint/evaluation directories under `logs/bg_pdr_fm/aaai27/formal/`.
Use `formal_smooth_dix.yaml`, `formal_sv_inv_net.yaml`,
`formal_velocity_gan.yaml`, `formal_conditional_ddpm.yaml`,
`formal_adapted_gfi_multimodal.yaml`, and
`formal_adapted_auto_linear_multimodal.yaml` for paper-facing adapted
baselines. `launch_aaai27_formal.py` launches the learned baselines by default;
run Smooth-Dix through the evaluator because it has no training phase. Use the
matched, strong, and no-gate formal configs for internal ablations only.

# AAAI Code Release Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-driven AAAI release surface for `bg_pdr_fm` that provides direct, testable evidence for the applicable reproducibility checklist items without changing unsupported paper claims.

**Architecture:** Keep `bg_pdr_fm` as the single model/data/evaluation implementation. Add a small `reproducibility/` layer for environment, manifest validation, and provenance; add thin `scripts/aaai27/` wrappers; add canonical relative-path configs under `bg_pdr_fm/configs/release/aaai27/`; and leave historical experiment variants outside the default release manifest.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, Lightning 2.5.2, OmegaConf, NumPy, SciPy, scikit-image, LMDB, PyYAML, pytest, Bash, and the existing AAAI LaTeX toolchain.

---

### Task 1: Add Release Metadata and Source Boundaries

**Files:**
- Create: `reproducibility/README.md`
- Create: `reproducibility/requirements.txt`
- Create: `reproducibility/THIRD_PARTY_NOTICES.md`
- Create: `docs/paper/AAAI2027/code_release_map.md`
- Modify: `.gitignore`
- Test: `bg_pdr_fm/tests/test_release_manifest.py`

- [ ] **Step 1: Write the release metadata test**

Add a test that loads the release manifest path declared by `reproducibility/README.md`, asserts that every listed source path exists, and rejects paths below `_build_full`, `_build_submission`, `logs`, `checkpoints`, or a filename ending in `.local.yaml`.

```python
def test_release_manifest_contains_only_source_inputs():
    manifest = load_release_manifest()
    assert manifest["version"] == 1
    assert manifest["entrypoints"]
    for path in manifest["files"]:
        assert Path(path).is_file(), path
        assert not any(part in {"logs", "checkpoints", "_build_full", "_build_submission"} for part in Path(path).parts)
        assert not path.endswith(".local.yaml")
```

- [ ] **Step 2: Run the new test and verify it fails**

Run: `pytest -q bg_pdr_fm/tests/test_release_manifest.py`

Expected: FAIL because the manifest loader and release metadata do not exist.

- [ ] **Step 3: Add the source boundary and dependency documents**

Create a machine-readable `reproducibility/release_manifest.yaml` with `version: 1`, the canonical train/evaluate/smoke/validate entrypoints, the core package modules, the release configs, the metric implementation, and the focused tests. List external baseline bridges under `external_sources` with their source commit fields instead of treating them as project-owned code. Document the exact Python/library versions from Appendix `Implementation and Evaluation Protocols` in `requirements.txt`; keep the ROCm/PyTorch wheel selection as an installation note rather than an unverifiable index URL. Document the distinction between project-owned code and external bridge code in `THIRD_PARTY_NOTICES.md`, and map each canonical module to paper sections/equations and checklist items in `code_release_map.md`.

Add these `.gitignore` rules without removing existing rules:

```gitignore
reproducibility/generated/
release_outputs/
bg_pdr_fm/configs/**/*.local.yaml
docs/paper/AAAI2027/_build_submission/
docs/paper/AAAI2027/_build_full/
_build_full/
```

- [ ] **Step 4: Run the manifest test and inspect the source-only diff**

Run: `pytest -q bg_pdr_fm/tests/test_release_manifest.py`

Expected: PASS, with the manifest enumerating only versioned source/config/test files.

- [ ] **Step 5: Commit**

```bash
git add .gitignore reproducibility docs/paper/AAAI2027/code_release_map.md bg_pdr_fm/tests/test_release_manifest.py
git commit -m "chore: define AAAI release source boundaries"
```

### Task 2: Make Seed and Provenance Handling Explicit

**Files:**
- Create: `bg_pdr_fm/reproducibility.py`
- Modify: `bg_pdr_fm/training/train_bg_pdr_fm.py:67-77,171-203,325-335`
- Modify: `bg_pdr_fm/data/datasets.py:123-183`
- Test: `bg_pdr_fm/tests/test_reproducibility.py`
- Test: `bg_pdr_fm/tests/test_condition_contract_protocol.py:76-116`

- [ ] **Step 1: Write deterministic seed and path-validation tests**

Cover three contracts: two calls with the same training seed reproduce Python/NumPy/Torch draws and model initialization; a release config with an absolute data/checkpoint path is rejected; and split seed/well seed remain independent from training seed.

```python
def test_seed_everything_replays_all_local_rngs():
    seed_everything(2027)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(2027)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
```

- [ ] **Step 2: Run the tests and verify the missing helper failure**

Run: `pytest -q bg_pdr_fm/tests/test_reproducibility.py`

Expected: FAIL with an import error for `bg_pdr_fm.reproducibility`.

- [ ] **Step 3: Implement the shared reproducibility helpers**

Implement:

```python
DEFAULT_TRAINING_SEED = 2027

def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    lightning.seed_everything(seed, workers=True)
    return seed
```

Add `assert_release_paths(conf, repo_root)` to reject absolute `data.*`, `training.checkpoint*`, and `evaluation.*` paths in release profiles, and `resolved_config_hash(conf)` to hash the OmegaConf-resolved YAML. Keep `data.split_seed` and `data.well_seed` as separate fields. Update `apply_training_seed` to delegate to `seed_everything`, and pass a `torch.Generator` seeded from `training.seed` to the shuffled training DataLoader. Keep validation/test loaders unshuffled.

- [ ] **Step 4: Run deterministic tests and the existing protocol tests**

Run: `pytest -q bg_pdr_fm/tests/test_reproducibility.py bg_pdr_fm/tests/test_condition_contract_protocol.py -k 'seed or split or well'`

Expected: PASS; the tests must show that identical seeds reproduce draws and that split/well membership remains stable.

- [ ] **Step 5: Commit**

```bash
git add bg_pdr_fm/reproducibility.py bg_pdr_fm/training/train_bg_pdr_fm.py bg_pdr_fm/data/datasets.py bg_pdr_fm/tests/test_reproducibility.py bg_pdr_fm/tests/test_condition_contract_protocol.py
git commit -m "feat: make AAAI randomness and provenance explicit"
```

### Task 3: Add Canonical Paper Configurations

**Files:**
- Create: `bg_pdr_fm/configs/release/aaai27/base.yaml`
- Create: `bg_pdr_fm/configs/release/aaai27/train.yaml`
- Create: `bg_pdr_fm/configs/release/aaai27/smoke.yaml`
- Create: `bg_pdr_fm/configs/release/aaai27/evaluate.yaml`
- Create: `bg_pdr_fm/configs/release/aaai27/ablation.yaml`
- Modify: `bg_pdr_fm/training/benchmark_config.py:11-21`
- Test: `bg_pdr_fm/tests/test_release_config.py`

- [ ] **Step 1: Write config contract tests**

Assert that the resolved `train.yaml` has the paper values: eight OpenFWI subsets, `[0.7, 0.2, 0.1]`, `split_seed: 42`, `well_seed: 1234`, `training.seed: 2027`, identity codec, direct 192-channel background U-Net, 128-channel residual FiLM U-Net, `residual_num_inference_steps: 50`, `max_epochs: 100`, `batch_size: 64`, `precision: bf16-mixed`, and `joint.lambda_contrastive: 0.01`. Assert that no resolved path is absolute and that the smoke config changes only bounded execution fields.

- [ ] **Step 2: Run the config tests and verify they fail**

Run: `pytest -q bg_pdr_fm/tests/test_release_config.py`

Expected: FAIL because the release config directory is absent.

- [ ] **Step 3: Create the relative-path config family**

Base the values on `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml`, but replace machine paths with `data/openfwi` and `data/openfwi_lmdb`, add `training.seed: 2027`, and use stable `release_outputs/aaai27/...` directories. `train.yaml` must use `training.stage: joint_full`, identity codec, `background_backend: unet_direct`, `background_hidden_channels: 192`, `residual_backend: unet_fm_film`, `residual_backend_hidden_channels: 128`, `residual_num_inference_steps: 50`, `quality_gate.mode: none`, and the paper loss weights. `smoke.yaml` must extend the same base and set one epoch, one train batch, one validation batch, zero workers, and two inference steps. `evaluate.yaml` must require checkpoints and process `split: test` with `shuffle=False`. `ablation.yaml` must expose the paper's concat/full-field and prediction-consistency controls without changing the main config.

Update `load_benchmark_config` to resolve relative `extends` paths from the config directory and expose the resolved config hash used by the manifest. Do not add stage-specific parameter overrides to wrapper scripts.

- [ ] **Step 4: Run config and compile checks**

Run: `pytest -q bg_pdr_fm/tests/test_release_config.py`

Expected: PASS with the exact paper values and no absolute paths.

- [ ] **Step 5: Commit**

```bash
git add bg_pdr_fm/configs/release bg_pdr_fm/training/benchmark_config.py bg_pdr_fm/tests/test_release_config.py
git commit -m "feat: add canonical AAAI paper configs"
```

### Task 4: Add Thin Reproducibility Entry Points

**Files:**
- Create: `scripts/aaai27/prepare_openfwi.py`
- Create: `scripts/aaai27/train.py`
- Create: `scripts/aaai27/evaluate.py`
- Create: `scripts/aaai27/smoke.py`
- Create: `scripts/aaai27/validate.py`
- Create: `scripts/aaai27/README.md`
- Modify: `bg_pdr_fm/data/openfwi_lmdb.py:1-193`
- Test: `bg_pdr_fm/tests/test_release_entrypoints.py`

- [ ] **Step 1: Write entrypoint tests**

Test each wrapper with `--help`, test that `train.py --config ... --dry-run` resolves the release config without calling Lightning, and test that `smoke.py` uses the synthetic dataset when no OpenFWI root is present.

- [ ] **Step 2: Run the tests and verify missing-entrypoint failures**

Run: `pytest -q bg_pdr_fm/tests/test_release_entrypoints.py`

Expected: FAIL because `scripts/aaai27` does not exist.

- [ ] **Step 3: Implement thin wrappers without duplicated training logic**

Each wrapper must add the repository root to `sys.path`, parse only `--config`, `--output`, `--data-root`, `--dry-run`, and `--force` options, then call package functions. `train.py` calls `train_bg_pdr_fm.main`; `evaluate.py` calls `evaluate_bg_pdr_fm.run_evaluation`; `smoke.py` loads `smoke.yaml` and calls the same training/evaluation APIs; `validate.py` calls `reproducibility.validate_release`; `prepare_openfwi.py` validates the four required modalities, builds or checks the LMDB manifest, and writes counts, split seed, well seed, and input hashes. No wrapper may assign batch size, epoch count, precision, or checkpoint paths.

Document exact commands, expected output directories, the required OpenFWI source layout, and the CPU synthetic smoke command in `scripts/aaai27/README.md`.

- [ ] **Step 4: Run wrapper checks**

Run: `python scripts/aaai27/train.py --help`; `python scripts/aaai27/evaluate.py --help`; `python scripts/aaai27/smoke.py --help`; `pytest -q bg_pdr_fm/tests/test_release_entrypoints.py`

Expected: all help commands exit 0 and tests pass without a real dataset.

- [ ] **Step 5: Commit**

```bash
git add scripts/aaai27 bg_pdr_fm/data/openfwi_lmdb.py bg_pdr_fm/tests/test_release_entrypoints.py
git commit -m "feat: add AAAI reproducibility entrypoints"
```

### Task 5: Record Evaluation Provenance and Validate Metrics

**Files:**
- Modify: `bg_pdr_fm/evaluation/benchmark_metrics.py:38-130`
- Modify: `bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py:209-364`
- Create: `bg_pdr_fm/evaluation/run_manifest.py`
- Test: `bg_pdr_fm/tests/test_run_manifest.py`
- Test: `bg_pdr_fm/tests/test_theory_diagnostics.py:1-127`

- [ ] **Step 1: Write manifest and metric-schema tests**

Require a manifest with `git_commit`, `config_sha256`, `training_seed`, `split_seed`, `well_seed`, `checkpoint_sha256`, `trajectory_count`, `evaluated_records`, `split`, `shuffle`, `literature_transcribed`, and a metric-name list. Add tests rejecting duplicate/extra/missing record IDs and non-finite metric values.

- [ ] **Step 2: Run the tests and verify missing-manifest failures**

Run: `pytest -q bg_pdr_fm/tests/test_run_manifest.py`

Expected: FAIL because the manifest writer and metric schema validator are absent.

- [ ] **Step 3: Implement the provenance path**

Add `write_run_manifest(output_dir, conf, checkpoint_paths, record_ids, *, trajectory_count, literature_transcribed)` that hashes the resolved config and each existing checkpoint, records the current Git commit, verifies unique record IDs, and writes `run_manifest.json` atomically. Make `run_evaluation` call it after all records are processed and before writing aggregate summaries. Add `METRIC_SCHEMA = ("mae", "rmse", "ssim", "mae_l", "mae_h")`, finite checks, and explicit sample-count fields to `benchmark_metrics.py`. Preserve the paper's per-sample RMSE and global-SSIM definitions.

- [ ] **Step 4: Run focused evaluator tests**

Run: `pytest -q bg_pdr_fm/tests/test_run_manifest.py bg_pdr_fm/tests/test_theory_diagnostics.py`

Expected: PASS; synthetic evaluation writes a manifest with one trajectory, the expected record count, `shuffle: false`, and finite metrics.

- [ ] **Step 5: Commit**

```bash
git add bg_pdr_fm/evaluation/benchmark_metrics.py bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py bg_pdr_fm/evaluation/run_manifest.py bg_pdr_fm/tests/test_run_manifest.py bg_pdr_fm/tests/test_theory_diagnostics.py
git commit -m "feat: record reproducible evaluation provenance"
```

### Task 6: Reconcile Code Comments, Inference Boundaries, and Paper Build Inputs

**Files:**
- Modify: `bg_pdr_fm/models/encoder.py:26-188`
- Modify: `bg_pdr_fm/lightning/stage_losses.py:132-361`
- Modify: `bg_pdr_fm/models/generators.py:152-452`
- Modify: `bg_pdr_fm/training/train_bg_pdr_fm.py:67-77,325-335`
- Modify: `bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py:260-364`
- Create: `docs/paper/AAAI2027/paper_submission.tex`
- Modify: `docs/paper/AAAI2027/build.sh`
- Test: `bg_pdr_fm/tests/test_inference_contract.py`
- Test: `bg_pdr_fm/tests/test_paper_build_inputs.py`

- [ ] **Step 1: Write inference and build-input tests**

Test that `predict_batch` accepts observed modality tensors and masks but rejects target/anchor keys, that training-only Haar anchors are not read in the inference branch, and that both `paper.tex` and `paper_submission.tex` exist with the wrapper defining `\aaaiwithoutappendix` before input.

- [ ] **Step 2: Run the tests and verify the missing wrapper/contract failure**

Run: `pytest -q bg_pdr_fm/tests/test_inference_contract.py bg_pdr_fm/tests/test_paper_build_inputs.py`

Expected: FAIL for the deleted wrapper or any current target-leakage path.

- [ ] **Step 3: Add equation-linked comments and restore the wrapper**

Add short module/function comments, not narration, at the implementation boundaries:

```python
# Paper Eq. (2): C_bg is the deterministic background interface; C_str is the residual interface.
# Paper Eqs. (4)-(6): use the detached B_hat for R_hatB supervision, context, and composition.
```

Document the exact paper section/equation in `encoder.py`, `stage_losses.py`, and `generators.py`; document that `A_bg/A_str` are training-only and that `V` never enters inference. Restore `paper_submission.tex` as a minimal wrapper that sets `\aaaiwithoutappendixtrue`, inputs `paper.tex`, and leaves bibliography placement controlled by the existing source. Update `build.sh` to resolve paths relative to its own directory and to fail with a clear message when the configured `pdflatex` binary is absent.

- [ ] **Step 4: Run contract and source checks**

Run: `pytest -q bg_pdr_fm/tests/test_inference_contract.py bg_pdr_fm/tests/test_paper_build_inputs.py`; `python -m compileall -q bg_pdr_fm scripts/aaai27`

Expected: PASS and no unresolved import or target-leakage failure.

- [ ] **Step 5: Commit**

```bash
git add bg_pdr_fm/models/encoder.py bg_pdr_fm/lightning/stage_losses.py bg_pdr_fm/models/generators.py bg_pdr_fm/training/train_bg_pdr_fm.py bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py docs/paper/AAAI2027/paper_submission.tex docs/paper/AAAI2027/build.sh bg_pdr_fm/tests/test_inference_contract.py bg_pdr_fm/tests/test_paper_build_inputs.py
git commit -m "fix: align release inference and paper build contracts"
```

### Task 7: Add Release Validator and Checklist Evidence Report

**Files:**
- Create: `reproducibility/validate_release.py`
- Create: `reproducibility/checklist_evidence.yaml`
- Modify: `reproducibility/release_manifest.yaml`
- Modify: `docs/paper/AAAI2027/ReproducibilityChecklist.tex`
- Modify: `bg_pdr_fm/README.md`
- Test: `bg_pdr_fm/tests/test_checklist_evidence.py`

- [ ] **Step 1: Write evidence tests**

Assert that every code-related checklist item has a path and a verification command, that the report distinguishes repository facts from publication-dependent claims, and that the checklist contains no instructional `Type your response here` text in the answer section.

- [ ] **Step 2: Run the tests and verify the missing validator failure**

Run: `pytest -q bg_pdr_fm/tests/test_checklist_evidence.py`

Expected: FAIL because the validator and evidence report do not exist.

- [ ] **Step 3: Implement the validator and evidence report**

`validate_release.py` must check the manifest, canonical configs, dependency file, wrapper entrypoints, source imports, seed fields, metric schema, and generated run-manifest schema. It must emit a concise table with `status`, `evidence_path`, and `command` for checklist items 4.2 through 4.14. Mark license/publication availability and literature rerun status as conditional when the repository cannot prove them. Update `bg_pdr_fm/README.md` to make the canonical commands the first-running path and move historical experiments below an explicit internal-research section. Update checklist answers only where the new evidence supports them; retain `partial`/`no` for hyperparameter search, independent run variation, and statistical significance where the paper explicitly limits those claims.

- [ ] **Step 4: Run the release validator and evidence tests**

Run: `python reproducibility/validate_release.py`; `pytest -q bg_pdr_fm/tests/test_checklist_evidence.py`

Expected: validator exits 0 for the source-only synthetic release and writes `reproducibility/generated/checklist_evidence.json` (ignored by Git).

- [ ] **Step 5: Commit**

```bash
git add reproducibility/validate_release.py reproducibility/checklist_evidence.yaml reproducibility/release_manifest.yaml docs/paper/AAAI2027/ReproducibilityChecklist.tex bg_pdr_fm/README.md bg_pdr_fm/tests/test_checklist_evidence.py
git commit -m "docs: connect release evidence to reproducibility checklist"
```

### Task 8: Remove Release-Branch Build Noise and Run the Full Gate

**Files:**
- Delete from the release branch: `_build_full/`, `docs/paper/AAAI2027/_build_full/`, `docs/paper/AAAI2027/_build_submission/`
- Delete from the release branch: machine-local `bg_pdr_fm/configs/experiments/aaai27/*.local.yaml`
- Modify: `reproducibility/release_manifest.yaml`
- Test: repository-wide focused test commands below

- [ ] **Step 1: Write the generated-artifact guard before cleanup**

Extend `test_release_manifest.py` to assert that no tracked release-manifest path is a generated build artifact, checkpoint, log, frame-sequence image, or local config.

- [ ] **Step 2: Run the guard against the checkpoint branch**

Run: `pytest -q bg_pdr_fm/tests/test_release_manifest.py`

Expected: FAIL because the checkpoint intentionally contains generated artifacts.

- [ ] **Step 3: Remove only validated generated targets**

Use explicit `git rm -r` targets from the files list above, keep paper source figures and editable assets required by `paper.tex`, and leave the recovery commit untouched. Do not remove checkpoints or external source bridges that the manifest identifies as provenance inputs.

- [ ] **Step 4: Run the complete verification gate**

Run:

```bash
python -m compileall -q bg_pdr_fm scripts/aaai27
pytest -q bg_pdr_fm/tests/test_release_manifest.py bg_pdr_fm/tests/test_reproducibility.py bg_pdr_fm/tests/test_release_config.py bg_pdr_fm/tests/test_release_entrypoints.py bg_pdr_fm/tests/test_run_manifest.py bg_pdr_fm/tests/test_inference_contract.py bg_pdr_fm/tests/test_paper_build_inputs.py bg_pdr_fm/tests/test_checklist_evidence.py
python scripts/aaai27/smoke.py --config bg_pdr_fm/configs/release/aaai27/smoke.yaml
python scripts/aaai27/validate.py --config bg_pdr_fm/configs/release/aaai27/train.yaml
```

Expected: compile succeeds, all focused tests pass, synthetic smoke writes finite metrics and a run manifest, and validation exits 0. If the TeX toolchain is installed, also run `docs/paper/AAAI2027/build.sh full` and `docs/paper/AAAI2027/build.sh submission`; otherwise report the missing toolchain without claiming a successful PDF build.

- [ ] **Step 5: Commit the cleanup and verification artifacts**

```bash
git add -A
git commit -m "chore: finalize AAAI reproducibility release surface"
```

After this commit, run `git status --short --branch`, `git diff HEAD^ --check -- bg_pdr_fm scripts reproducibility docs/paper/AAAI2027`, and report the exact test counts, skipped real-data tests, manifest path, and any checklist items that remain publication-dependent.

# Table 2 Qualitative Visualization Report

## Protocol

- Exact global held-out records audited: `33600`.
- Criterion-selected advantage cases are retained only when fixed-seed re-inference preserves their positive formal margin.
- Per-subset cases maximize the immutable formal SSIM margin and retain that formal winner even if its re-rendered margin changes under deterministic PD-BG-RFM sampling.
- Baseline predictions use each checkpoint's native formal normalization; plotted velocities are converted to m/s.
- Cross-method rendered margins use a common OpenFWI [-1, 1] metric coordinate after physical conversion.
- PD-BG-RFM uses a deterministic seed derived from `(2027, dataset_name, source_sample_index)`.
- Shared display ranges: velocity `[1500, 4500]` m/s; absolute error `(0.0, 750.0)` m/s.
- Observed selected-panel absolute-error range is `(0.0, 3225.500488)` m/s; values above the display maximum are clipped for visibility.
- Velocity maps use the fixed `jet` colormap; absolute-error maps use the sequential `inferno` colormap; every velocity panel displays its per-image SSIM.
- Spatial axes use `x/z grid index` with ticks 0, 35, and 69; no physical-distance unit is implied.

## Automatically Selected Cases

### Criterion-Selected Advantage

- `FlatFaultA/22085 (dataset_id=4)`: criterion=margin_l, margin_l=+0.004111, rendered_margin_l=+0.009909, margin_h=-0.001723, rendered_margin_h=+0.000931, margin_s=+0.003394, rendered_margin_s=+0.000396.
- `CurveVelB/23726 (dataset_id=3)`: criterion=margin_h, margin_l=-0.007420, rendered_margin_l=-0.009492, margin_h=+0.002498, rendered_margin_h=+0.008558, margin_s=+0.031180, rendered_margin_s=+0.005482.
- `CurveVelA/3929 (dataset_id=2)`: criterion=margin_s, margin_l=-0.003772, rendered_margin_l=+0.001068, margin_h=-0.001139, rendered_margin_h=-0.000298, margin_s=+0.109192, rendered_margin_s=+0.000191.

### Per-Subset Maximum Formal SSIM Margin

- `CurveFaultA/26193 (dataset_id=6)`: criterion=margin_s_max, margin_l=-0.004294, rendered_margin_l=+0.005052, margin_h=-0.004199, rendered_margin_h=-0.001361, margin_s=+0.094422, rendered_margin_s=+0.000341.
- `CurveFaultB/7449 (dataset_id=7)`: criterion=margin_s_max, margin_l=-0.008979, rendered_margin_l=-0.004705, margin_h=-0.006466, rendered_margin_h=+0.001725, margin_s=+0.088499, rendered_margin_s=+0.000199.
- `CurveVelA/3929 (dataset_id=2)`: criterion=margin_s_max, margin_l=-0.003772, rendered_margin_l=+0.001068, margin_h=-0.001139, rendered_margin_h=-0.000298, margin_s=+0.109192, rendered_margin_s=+0.000191.
- `CurveVelB/26377 (dataset_id=3)`: criterion=margin_s_max, margin_l=-0.014747, rendered_margin_l=-0.008814, margin_h=-0.003693, rendered_margin_h=-0.000445, margin_s=+0.190374, rendered_margin_s=-0.000489.
- `FlatFaultA/10401 (dataset_id=4)`: criterion=margin_s_max, margin_l=-0.014316, rendered_margin_l=-0.003108, margin_h=-0.004156, rendered_margin_h=+0.000147, margin_s=+0.053382, rendered_margin_s=+0.000252.
- `FlatFaultB/21686 (dataset_id=5)`: criterion=margin_s_max, margin_l=-0.008023, rendered_margin_l=-0.001189, margin_h=-0.003438, rendered_margin_h=-0.000861, margin_s=+0.092348, rendered_margin_s=-0.000235.
- `FlatVelA/101 (dataset_id=0)`: criterion=margin_s_max, margin_l=-0.002845, rendered_margin_l=-0.002626, margin_h=-0.002233, rendered_margin_h=+0.001712, margin_s=+0.049086, rendered_margin_s=+0.072839.
- `FlatVelB/29501 (dataset_id=1)`: criterion=margin_s_max, margin_l=-0.014551, rendered_margin_l=-0.015836, margin_h=-0.002678, rendered_margin_h=-0.002914, margin_s=+0.169346, rendered_margin_s=-0.001137.

### Failure

- `CurveFaultB/7146 (dataset_id=7)`: criterion=margin_l, margin_l=-0.074222, rendered_margin_l=-0.056868, margin_h=-0.016482, rendered_margin_h=-0.011534, margin_s=-0.066386, rendered_margin_s=-0.065252.
- `CurveVelB/1645 (dataset_id=3)`: criterion=margin_h, margin_l=-0.066328, rendered_margin_l=-0.069267, margin_h=-0.035038, rendered_margin_h=-0.032189, margin_s=-0.160093, rendered_margin_s=-0.319281.
- `FlatFaultA/3215 (dataset_id=4)`: criterion=margin_s, margin_l=-0.013224, rendered_margin_l=-0.005903, margin_h=-0.001836, rendered_margin_h=-0.000518, margin_s=-0.832114, rendered_margin_s=-0.556588.

## Interpretation Limit

The criterion-selected and per-subset maximum-formal-SSIM-margin panels are deliberate case selections, not claims of aggregate superiority. The per-subset rule retains the maximum formal margin even when it is non-positive; rendered margins are reported for audit, and the failure panel is retained alongside it.

## Figure Files

- `table2_advantage_cases`: `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_advantage_cases.pdf` and `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_advantage_cases.png`.
- `table2_representative_cases`: `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases.pdf` and `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases.png`.
- `table2_failure_cases`: `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_failure_cases.pdf` and `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_failure_cases.png`.
- `table2_representative_cases_1`: `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases_1.pdf` and `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases_1.png`.
- `table2_representative_cases_2`: `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases_2.pdf` and `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/table2_representative_cases_2.png`.
- `standalone_velocity`: `56` coordinate-free PNGs in `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/standalone_velocity` with metadata in `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/standalone_velocity_manifest.csv`.
- `standalone_error`: `48` coordinate-free PNGs in `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/standalone_error` with metadata in `/public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027/figures/table2_qualitative/standalone_error_manifest.csv`.

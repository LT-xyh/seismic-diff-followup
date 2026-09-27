# AAAI 2027 Paper Build

This directory keeps two builds from the same source:

- `pd-bg-rfm.tex`: current editable manuscript with the in-PDF technical appendix.
- `paper_pd-bg-rfm_AAAI2027.pdf`: retained submission PDF snapshot under `docs/paper/aaaii/`.

Build commands:

```bash
cd /public/home/xuyinghao/workspace/seismic-diff/docs/paper/AAAI2027
./build.sh submission
./build.sh full
```

Outputs:

- `_build_pd_bg_rfm/pd-bg-rfm.pdf`: local build output when the native TeX toolchain is available.
- `_build_full/paper.pdf`: working/reference PDF with appendix.

Do not manually crop the full PDF for submission unless you have verified that references remain in the submitted PDF. The wrapper build is safer because it removes the appendix before bibliography placement.

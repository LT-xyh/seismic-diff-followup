# Third-Party Notices

This file records provenance for source-derived baseline components. It is not
a project license and does not grant rights beyond the upstream notices.

## LANL OpenFWI Network Definitions

- Release path: `bg_pdr_fm/external_baselines/openfwi_official.py`
- Upstream: LANL OpenFWI, `network.py` and `utils.py`
- Source commit: `48754806b7b4c5877259c6b958a87f4513fcdc0b`
- Status: source-derived architecture definitions used by the adapted
  InversionNet, VelocityGAN, and UPFWI bridges.

The source file retains the copyright and U.S. Government license notice from
the upstream material. Do not remove that header or apply a project license to
the copied source without confirming compatibility with the upstream terms.

## Project-Owned Adapters

`inversion_net.py`, `velocity_gan.py`, `upfwi.py`, and `common.py` adapt the
five-channel multimodal protocol around the OpenFWI network definitions. They
are project-owned glue code, but their upstream architecture dependency remains
the LANL OpenFWI source listed above.

## Other Baseline Bridges

The `auto_linear/` and `gfi/` bridge directories are not part of the canonical
PD-BG-RFM training path. Their source provenance and license must be verified
against the exact release archive before public redistribution. Until that
review is complete, their `source_commit` is intentionally recorded as
`unresolved` in the release manifest and they must not be relicensed as project
code.

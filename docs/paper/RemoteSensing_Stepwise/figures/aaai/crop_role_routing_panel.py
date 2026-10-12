#!/usr/bin/env python3
"""Produce a compact main-text Figure 4 from the unmodified appendix panel (d).

This is a pixel-exact rectangular crop with minimal white padding. The original
appendix_full_condition_diagnostics.png is READ ONLY and must remain unmodified.
Only the redundant four-panel title "(d) Role-specific reliability routing"
and external blank space are excluded. All heatmap pixels, tick labels, values,
color bars, axis titles, subset names, and role-specific titles remain intact.
"""
from pathlib import Path
import hashlib
from PIL import Image, ImageOps, ImageChops

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "appendix_full_condition_diagnostics.png"
DEST = HERE / "role_routing_compact.png"
SOURCE_SHA256 = "03c60ef77943a887e783ec2f6b1418c9f878c45c02a5c49553f3ad1c6613ded4"
SOURCE_SIZE = (2561, 1545)
CROP = (12, 998, 2520, 1518)       # left, top, right, bottom in original pixels
PADDING = (0, 14, 0, 8)           # readability padding; no data is repainted

def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise SystemExit(f"Source artwork hash mismatch: {source_hash}")
    with Image.open(SOURCE) as im:
        if im.size != SOURCE_SIZE:
            raise SystemExit(f"Source image dimensions changed: {im.size}")
        # Cropping retains every source pixel within the scientific panel.
        panel = im.crop(CROP).convert("RGB")
        figure = ImageOps.expand(panel, border=PADDING, fill="white")
    figure.save(DEST, format="PNG", optimize=True)
    with Image.open(DEST) as im:
        assert im.size == (CROP[2]-CROP[0]+PADDING[0]+PADDING[2],
                           CROP[3]-CROP[1]+PADDING[1]+PADDING[3])
        output_panel = im.crop((PADDING[0],PADDING[1],
                               PADDING[0]+panel.width,PADDING[1]+panel.height))
        if ImageChops.difference(output_panel, panel).getbbox() is not None:
            raise SystemExit("Figure data pixels changed after cropping")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("Original appendix source mutated")
    print("FIGURE4_COMPACT_CROP_PASS", DEST, "pixels", figure.size,
          "source_sha256", SOURCE_SHA256,
          "output_sha256", hashlib.sha256(DEST.read_bytes()).hexdigest())

if __name__ == "__main__":
    main()

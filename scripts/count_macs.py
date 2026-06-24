"""Valid layer-wise MAC / FLOP counter for ECA-LDNet (CPU-only, no training).

Computes multiply-accumulate operations analytically from the built Keras model
by walking every layer and applying the standard per-layer MAC formulas:

  Conv2D                : H_out * W_out * C_out * (C_in/groups) * kH * kW
  DepthwiseConv2D       : H_out * W_out * C_in * depth_mult * kH * kW
  SeparableConv2D       : depthwise + pointwise
  Conv1D (ECA)          : L_out * C_out * C_in * k
  Dense                 : (prod(out_dims_except_batch_last)) * in_features * units
  (BN/activation/pool/add/concat/dropout/upsample contribute ~0 MACs)

FLOPs are reported as 2 x MACs (one multiply + one add per MAC), the convention
used by FFA-Net / DehazeFormer when they quote GFLOPs.

Usage:
    python scripts/count_macs.py            # 256x256, default
    python scripts/count_macs.py --size 256
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _spatial(shape):
    # shape like (None, H, W, C) -> (H, W); robust to missing dims
    dims = [d for d in shape[1:-1] if d is not None]
    return dims


def layer_macs(layer) -> int:
    """Analytic MACs for one Keras layer from its config + output shape."""
    cls = layer.__class__.__name__
    try:
        out_shape = layer.output.shape
    except Exception:
        return 0

    if cls == "Conv2D":
        H, W = _spatial(out_shape)
        c_out = int(out_shape[-1])
        kH, kW = layer.kernel_size
        c_in = int(layer.input.shape[-1])
        groups = getattr(layer, "groups", 1) or 1
        return H * W * c_out * (c_in // groups) * kH * kW

    if cls == "DepthwiseConv2D":
        H, W = _spatial(out_shape)
        kH, kW = layer.kernel_size
        c_in = int(layer.input.shape[-1])
        dm = getattr(layer, "depth_multiplier", 1)
        return H * W * c_in * dm * kH * kW

    if cls == "SeparableConv2D":
        H, W = _spatial(out_shape)
        kH, kW = layer.kernel_size
        c_in = int(layer.input.shape[-1])
        c_out = int(out_shape[-1])
        dm = getattr(layer, "depth_multiplier", 1)
        depthwise = H * W * c_in * dm * kH * kW
        pointwise = H * W * c_out * (c_in * dm)
        return depthwise + pointwise

    if cls == "Conv1D":
        sp = _spatial(out_shape)
        L = sp[0] if sp else 1
        c_out = int(out_shape[-1])
        (k,) = layer.kernel_size
        c_in = int(layer.input.shape[-1])
        groups = getattr(layer, "groups", 1) or 1
        return L * c_out * (c_in // groups) * k

    if cls == "Dense":
        units = int(layer.units)
        in_features = int(layer.input.shape[-1])
        # Dense applies per remaining (non-feature) position
        positions = _prod([d for d in out_shape[1:-1] if d is not None]) or 1
        return positions * in_features * units

    # BatchNormalization, ReLU/activations, pooling, Add, Concatenate, Dropout,
    # UpSampling2D, Reshape, Multiply (gating), Lambda: negligible MACs.
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Analytic MAC counter for ECA-LDNet")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from eca_ldnet.model import build_eca_ldnet

    model = build_eca_ldnet(img_size=args.size)

    total = 0
    by_type: dict[str, int] = {}
    rows = []
    for lyr in model.layers:
        m = layer_macs(lyr)
        if m:
            total += m
            by_type[lyr.__class__.__name__] = by_type.get(lyr.__class__.__name__, 0) + m
            rows.append((lyr.name, lyr.__class__.__name__, m))

    print(f"Input resolution      : {args.size} x {args.size} x 3")
    print(f"Parameters            : {model.count_params():,}")
    print(f"Total MACs            : {total:,}")
    print(f"Total MACs (G)        : {total / 1e9:.3f} GMACs")
    print(f"Total FLOPs (2x MACs) : {2 * total / 1e9:.3f} GFLOPs")
    print("\nMACs by layer type:")
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<18} {v/1e9:7.3f} GMACs  ({100*v/total:5.1f}%)")

    print("\nTop 12 layers by MACs:")
    for name, cls, m in sorted(rows, key=lambda r: -r[2])[:12]:
        print(f"  {name:<28} {cls:<16} {m/1e6:8.2f} MMACs")


if __name__ == "__main__":
    main()

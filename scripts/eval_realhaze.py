"""No-reference real-haze evaluation (inference-only, no retraining, CPU-OK).

Runs the frozen ECA-LDNet checkpoint on real hazy images and reports
no-reference quality metrics (NIQE, BRISQUE) on hazy vs. dehazed pairs.

Two comparison modes are produced:
  * matched-256 : hazy and dehazed both scored at the model's native 256x256.
                  This isolates the network's effect from resampling.
  * native      : dehazed upsampled back to source resolution (the deployment
                  pipeline in eca_ldnet.inference.dehaze_file). NR metrics here
                  are dominated by the 256->full upscale and are NOT a fair
                  measure of the model itself -- reported only for awareness.

NIQE and BRISQUE use pyiqa (lower is better). FADE is NOT yet implemented in
this repo or in pyiqa; it requires a faithful port of Choi et al. 2015 with the
authors' fog-aware MVG parameters (still inference-only / no GPU).

Outputs (written under --out):
  * hazy256/, dehazed256/        : the scored 256x256 PNG pairs
  * realhaze_metrics.csv         : per-image + MEAN NIQE/BRISQUE (paste-ready)
  * fig_realhaze_qualitative.png : hazy-vs-dehazed grid for the paper figure

Usage:
    python -m scripts.eval_realhaze --model checkpoints/stage5_final_ema_model.keras \
        --hazy ../realhaze_eval/hazy --out ../realhaze_eval
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser(description="NR real-haze evaluation (inference only)")
    p.add_argument("--model", default="checkpoints/stage5_final_ema_model.keras")
    p.add_argument("--hazy", required=True, help="Directory of real hazy images")
    p.add_argument("--out", default="./realhaze_eval")
    args = p.parse_args()

    import cv2
    import numpy as np
    import pyiqa

    from eca_ldnet.data import preprocess_hazy
    from eca_ldnet.inference import dehaze_array
    from eca_ldnet.model import load_pretrained

    h256 = os.path.join(args.out, "hazy256")
    d256 = os.path.join(args.out, "dehazed256")
    os.makedirs(h256, exist_ok=True)
    os.makedirs(d256, exist_ok=True)

    model = load_pretrained(args.model)
    niqe = pyiqa.create_metric("niqe", device="cpu")
    bris = pyiqa.create_metric("brisque", device="cpu")

    def score(path: str):
        return float(niqe(path).item()), float(bris(path).item())

    rows = []
    pairs = []  # (name, hazy_rgb01, dehazed_rgb01) for the qualitative figure
    for hp in sorted(glob.glob(os.path.join(args.hazy, "*"))):
        name = os.path.splitext(os.path.basename(hp))[0]
        img = cv2.imread(hp)
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        hpath = os.path.join(h256, name + ".png")
        dpath = os.path.join(d256, name + ".png")
        cv2.imwrite(hpath, cv2.cvtColor((rgb * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
        pred = dehaze_array(model, preprocess_hazy(rgb))
        cv2.imwrite(dpath, cv2.cvtColor((pred * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
        hn, hb = score(hpath)
        dn, db = score(dpath)
        rows.append((name, hn, dn, hb, db))
        pairs.append((name, rgb, np.clip(pred, 0, 1)))
        print(f"{name:<14} NIQE {hn:5.2f} -> {dn:5.2f}  | BRISQUE {hb:6.2f} -> {db:6.2f}")

    if not rows:
        print("No images found in --hazy; nothing to do.")
        return

    mean_hn = st.mean(r[1] for r in rows)
    mean_dn = st.mean(r[2] for r in rows)
    mean_hb = st.mean(r[3] for r in rows)
    mean_db = st.mean(r[4] for r in rows)
    print("-" * 64)
    print(
        "MEAN          NIQE %5.2f -> %5.2f  | BRISQUE %6.2f -> %6.2f"
        % (mean_hn, mean_dn, mean_hb, mean_db)
    )

    # ---- CSV (paste-ready for the paper's real-haze table) ----
    import csv

    csv_path = os.path.join(args.out, "realhaze_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "niqe_hazy", "niqe_dehazed", "brisque_hazy", "brisque_dehazed"])
        w.writerows(rows)
        w.writerow(["MEAN", f"{mean_hn:.3f}", f"{mean_dn:.3f}", f"{mean_hb:.3f}", f"{mean_db:.3f}"])
    print(f"Saved {csv_path}")

    # ---- Qualitative figure (hazy top row, dehazed bottom row) for the paper ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(len(pairs), 6)  # cap to keep the figure legible
        sel = pairs[:n]
        fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.4))
        if n == 1:
            axes = axes.reshape(2, 1)
        for j, (name, hz, de) in enumerate(sel):
            axes[0, j].imshow(hz)
            axes[0, j].set_title(name, fontsize=8)
            axes[0, j].axis("off")
            axes[1, j].imshow(de)
            axes[1, j].axis("off")
        axes[0, 0].set_ylabel("Hazy input", fontsize=9)
        axes[1, 0].set_ylabel("ECA-LDNet", fontsize=9)
        # re-enable y-labels (axis('off') hides them) via text on the left margin
        fig.text(0.012, 0.74, "Hazy input", rotation=90, va="center", fontsize=9)
        fig.text(0.012, 0.27, "ECA-LDNet", rotation=90, va="center", fontsize=9)
        fig.tight_layout(rect=[0.03, 0, 1, 1])
        fig_path = os.path.join(args.out, "fig_realhaze_qualitative.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved {fig_path}")
    except Exception as e:  # matplotlib optional; metrics already saved
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()

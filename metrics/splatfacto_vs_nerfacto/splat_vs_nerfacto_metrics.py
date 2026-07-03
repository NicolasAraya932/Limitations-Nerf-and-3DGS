#!/usr/bin/env python3
"""Splatfacto vs Nerfacto PSNR, one figure per variant family.

Frozen-contract reconstruction PSNR (mean +/- std over seeds) for both scenes,
plotted in the shared paper style of this repo (see gamma_metrics.py /
height_metrics.py): Nerfacto = blue circle, Splatfacto = orange triangle;
Indoor = hollow marker, Outdoor = filled marker; undegraded ``original`` shown
as a horizontal GT reference per scene+method. One PNG per family.

Data: the long-format frozen-contract CSVs
(``family,x,label,method,psnr_mean,psnr_std,n_seeds``) copied into ``data/``.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# --- Style (matches gamma_metrics.py) ---
METHOD_COLOR = {"Nerfacto": "#1f77b4", "Splatfacto": "#ff7f0e"}
GT_COLOR = {"Nerfacto": "#0d3a66", "Splatfacto": "#b35600"}
METHOD_MARKER = {"Nerfacto": "o", "Splatfacto": "^"}

FAMILY_XLABEL = {
    "blur": "Blur level (%)",
    "gamma": "Percentage of images with gamma noise",
    "pose": "Pose noise magnitude",
    "hemisphere": "Hemisphere subset index",
    "height": "Height variant (ordinal)",
    "reduced_img": "Image-reduction factor",
}
# Families whose indoor/outdoor x-axes are not directly comparable: plot on an
# ordinal position axis (like height_metrics.normalize_levels).
ORDINAL_FAMILIES = {"height"}
FAMILY_ORDER = ["blur", "gamma", "pose", "hemisphere", "height", "reduced_img"]


def load():
    """Return {scene: {'Nerfacto': df, 'Splatfacto': df}} with numeric x."""
    out = {}
    for scene in ("indoor", "outdoor"):
        out[scene.capitalize()] = {}
        for method, tag in (("Nerfacto", "nerfacto"), ("Splatfacto", "splatfacto")):
            df = pd.read_csv(os.path.join(DATA, f"{scene}_{tag}.csv"))
            df["x"] = df["x"].astype(float)
            out[scene.capitalize()][method] = df
    return out


def series(df, family):
    """(x, mean, std) sorted by x for one family, excluding 'original'."""
    sub = df[df["family"] == family].sort_values("x")
    return sub["x"].to_numpy(), sub["psnr_mean"].to_numpy(), sub["psnr_std"].to_numpy()


def gt_value(df):
    row = df[df["family"] == "original"]
    return float(row["psnr_mean"].iloc[0]) if not row.empty else None


def plot_family(family, data):
    fig, ax = plt.subplots(figsize=(8, 6))
    combos = [("Outdoor", "Splatfacto"), ("Outdoor", "Nerfacto"),
              ("Indoor", "Nerfacto"), ("Indoor", "Splatfacto")]
    ordinal = family in ORDINAL_FAMILIES

    # Determine x positions. For ordinal families map each scene's sorted x to
    # 1..N so mismatched indoor/outdoor parameterizations line up.
    all_x = []
    for scene, method in combos:
        x, _, _ = series(data[scene][method], family)
        all_x.append(x)
    if ordinal:
        base_positions = {i: i + 1 for i in range(max(len(x) for x in all_x))}
        spacing = 1.0
    else:
        xs = sorted(set(np.concatenate(all_x))) if all_x else []
        spacing = min(np.diff(xs)) if len(xs) > 1 else 1.0

    ou = spacing * 0.08  # offset unit to separate overlapping series
    combo_offset = {("Outdoor", "Splatfacto"): -1.5 * ou,
                    ("Outdoor", "Nerfacto"): -0.5 * ou,
                    ("Indoor", "Nerfacto"): 0.5 * ou,
                    ("Indoor", "Splatfacto"): 1.5 * ou}

    lows, highs = [], []
    xtick_pos, xtick_lab = None, None
    for scene, method in combos:
        df = data[scene][method]
        x, m, s = series(df, family)
        if len(x) == 0:
            continue
        if ordinal:
            xpos = np.arange(1, len(x) + 1, dtype=float)
            if xtick_pos is None or len(xpos) > len(xtick_pos):
                xtick_pos = xpos.copy()
                xtick_lab = [str(i) for i in range(1, len(x) + 1)]
        else:
            xpos = x.copy()
            xtick_pos = np.array(sorted(set(np.concatenate(all_x))))
            xtick_lab = [f"{v:g}" for v in xtick_pos]
        xoff = xpos + combo_offset[(scene, method)]

        color = METHOD_COLOR[method]
        marker = METHOD_MARKER[method]
        mfc = "none" if scene == "Indoor" else color
        lows.extend(m - s)
        highs.extend(m + s)

        ax.plot(xoff, m, "-", color=color, alpha=0.2, zorder=1)
        ax.errorbar(xoff, m, yerr=s, marker=marker, linestyle="none", capsize=2,
                    markersize=10, color=color, markerfacecolor=mfc,
                    markeredgewidth=1.5, zorder=2)

    # GT reference lines (undegraded original), per scene+method.
    for scene, method in [("Indoor", "Nerfacto"), ("Indoor", "Splatfacto"),
                          ("Outdoor", "Nerfacto"), ("Outdoor", "Splatfacto")]:
        gt = gt_value(data[scene][method])
        if gt is None:
            continue
        ls = "--" if scene == "Indoor" else ":"
        ax.axhline(gt, color=GT_COLOR[method], linestyle=ls, linewidth=1.2,
                   alpha=0.7, zorder=0)
        lows.append(gt)
        highs.append(gt)

    ax.set_title("PSNR (dB) ↑", fontsize=16)
    ax.set_xlabel(FAMILY_XLABEL[family], fontsize=16)
    ax.set_ylabel("PSNR (dB)")
    if xtick_pos is not None:
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab)

    # Dual-axis look (right spine mirrors data range, no labels).
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])

    # Legend: 4 series + 4 GT lines, ordered like the repo's plots.
    legend_elems, legend_labels = [], []
    for scene, method in [("Indoor", "Splatfacto"), ("Outdoor", "Splatfacto"),
                          ("Indoor", "Nerfacto"), ("Outdoor", "Nerfacto")]:
        color = METHOD_COLOR[method]
        mfc = "none" if scene == "Indoor" else color
        legend_elems.append(Line2D([0], [0], marker=METHOD_MARKER[method],
                                   color=color, linestyle="None", markersize=8,
                                   markerfacecolor=mfc, markeredgewidth=1.2))
        legend_labels.append(f"{method} {scene}")
    for scene, method in [("Indoor", "Splatfacto"), ("Outdoor", "Splatfacto"),
                          ("Indoor", "Nerfacto"), ("Outdoor", "Nerfacto")]:
        ls = "--" if scene == "Indoor" else ":"
        legend_elems.append(Line2D([0], [0], color=GT_COLOR[method],
                                   linestyle=ls, linewidth=1.2))
        legend_labels.append(f"{method} GT {scene}")
    ax.legend(legend_elems, legend_labels, loc="best", fontsize=8, ncol=2,
              markerscale=0.9)

    ax.tick_params(axis="y", which="both", direction="in", left=True, right=True)
    ax.tick_params(axis="x", which="both", direction="in", bottom=True, top=True)
    ax2.tick_params(axis="y", which="both", direction="in", left=False, right=True)

    fig.tight_layout()
    out = os.path.join(HERE, f"{family}_splat_vs_nerfacto_PSNR.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    data = load()
    for family in FAMILY_ORDER:
        print("saved", plot_family(family, data))


if __name__ == "__main__":
    main()

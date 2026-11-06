#!/usr/bin/env python3
"""Splatfacto vs Nerfacto PSNR drawn at printed size, in three layout options.

Same data, colours and markers as splat_vs_nerfacto_metrics.py (Nerfacto = blue
circle, Splatfacto = orange triangle, Indoor hollow, Outdoor filled, undegraded
GT as a darker marker at the leftmost "GT" slot), but sized for the paper so
LaTeX does not rescale the text: each panel is 0.32 of the 516 pt IEEE text
width, Times-metric serif, 8 pt labels and 7 pt ticks, TrueType embedding. This
matches the IoU panels of Figs 11 and 13.

    A  six panels, each with its own 4-entry legend (GT explained in the caption)
    B  six panels without legend + one shared legend strip (legend.pdf)
    C  one 2x3 figure at full text width, common legend, shared y-axis

Outputs go to print/opt{A,B,C}/.

Usage:
    python3 splat_vs_nerfacto_print.py            # all three options
    python3 splat_vs_nerfacto_print.py --options C
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from splat_vs_nerfacto_metrics import (FAMILY_ORDER, FAMILY_XLABEL, GT_COLOR, HERE,
                                       METHOD_COLOR, METHOD_MARKER, ORDINAL_FAMILIES,
                                       gt_value, load, series)

TEXT_W_IN = 516.0 / 72.0          # IEEE two-column \textwidth
PANEL_W_IN = 0.32 * TEXT_W_IN     # subfigure width used in the tex
PANEL_ASPECT = 0.72               # same as the IoU panels
LABEL_FS = 8
TICK_FS = 7
TITLE = r"PSNR (dB) $\uparrow$"

FAMILY_NAME = {
    "blur": "Motion Blur",
    "gamma": "Gamma Noise",
    "pose": "Pose Noise",
    "hemisphere": "Hemispherical View Filtering",
    "reduced_img": "Image Subsampling",
    "height": "Elevation-Based Subsampling",
}
# Paper order of the subfigures (a)-(f).
PAPER_ORDER = ["blur", "gamma", "pose", "hemisphere", "reduced_img", "height"]
LEGEND_ORDER = [("Indoor", "Nerfacto"), ("Outdoor", "Nerfacto"),
                ("Indoor", "Splatfacto"), ("Outdoor", "Splatfacto")]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "Times"],
        "mathtext.fontset": "stix",
        "font.size": TICK_FS,
        "axes.titlesize": LABEL_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": TICK_FS,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.8,
        "errorbar.capsize": 1.5,
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def draw_family(ax, family: str, data: dict) -> tuple[float, float]:
    """Draw one family on ax; returns (low, high) of the plotted values."""
    combos = [("Outdoor", "Splatfacto"), ("Outdoor", "Nerfacto"),
              ("Indoor", "Nerfacto"), ("Indoor", "Splatfacto")]
    ordinal = family in ORDINAL_FAMILIES
    all_x = [series(data[sc][m], family)[0] for sc, m in combos]
    if ordinal:
        spacing = 1.0
    else:
        xs = sorted(set(np.concatenate(all_x)))
        spacing = min(np.diff(xs)) if len(xs) > 1 else 1.0
    ou = spacing * 0.08
    offset = {c: (k - 1.5) * ou for k, c in enumerate(combos)}
    gt_pos = 0.0 if ordinal else sorted(set(np.concatenate(all_x)))[0] - spacing

    lows, highs, ticks = [], [], None
    for scene, method in combos:
        df = data[scene][method]
        x, m, s = series(df, family)
        if len(x) == 0:
            continue
        xpos = np.arange(1, len(x) + 1, dtype=float) if ordinal else x.copy()
        if ticks is None or len(xpos) > len(ticks[0]):
            ticks = (xpos, [str(i) for i in range(1, len(x) + 1)] if ordinal
                     else [f"{v:g}" for v in xpos])
        color, marker = METHOD_COLOR[method], METHOD_MARKER[method]
        mfc = "white" if scene == "Indoor" else color
        ax.plot(xpos + offset[(scene, method)], m, "-", color=color, alpha=0.2, zorder=1)
        ax.errorbar(xpos + offset[(scene, method)], m, yerr=s, marker=marker,
                    linestyle="none", color=color, markerfacecolor=mfc, zorder=2)
        lows.extend(m - s)
        highs.extend(m + s)
        gt = gt_value(df)
        if gt is not None:
            ax.plot(gt_pos + offset[(scene, method)], gt, marker=marker, linestyle="none",
                    color=GT_COLOR[method],
                    markerfacecolor="white" if scene == "Indoor" else GT_COLOR[method],
                    zorder=3)
            lows.append(gt)
            highs.append(gt)
    ax.set_xticks(np.concatenate([[gt_pos], ticks[0]]))
    ax.set_xticklabels(["GT"] + list(ticks[1]))
    ax.set_xlabel(FAMILY_XLABEL[family])
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    return min(lows), max(highs)


def legend_handles(with_gt: bool):
    handles, labels = [], []
    for scene, method in LEGEND_ORDER:
        color = METHOD_COLOR[method]
        handles.append(Line2D([0], [0], marker=METHOD_MARKER[method], color=color,
                              linestyle="none",
                              markerfacecolor="white" if scene == "Indoor" else color))
        labels.append(f"{method} {scene}")
    if with_gt:
        for scene, method in LEGEND_ORDER:
            color = GT_COLOR[method]
            handles.append(Line2D([0], [0], marker=METHOD_MARKER[method], color=color,
                                  linestyle="none",
                                  markerfacecolor="white" if scene == "Indoor" else color))
            labels.append(f"{method} GT {scene}")
    return handles, labels


def option_a(data: dict, out_dir: str) -> None:
    """Six panels, each with a compact 4-entry legend between title and axes."""
    for family in FAMILY_ORDER:
        fig, ax = plt.subplots(figsize=(PANEL_W_IN, PANEL_W_IN * 0.86), constrained_layout=True)
        draw_family(ax, family, data)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(TITLE, pad=2)
        h, l = legend_handles(with_gt=False)
        fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
                   handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)
        fig.savefig(os.path.join(out_dir, f"{family}_PSNR.pdf"), bbox_inches="tight")
        plt.close(fig)


def option_b(data: dict, out_dir: str) -> None:
    """Six legend-free panels plus one legend strip spanning the text width."""
    for family in FAMILY_ORDER:
        fig, ax = plt.subplots(figsize=(PANEL_W_IN, PANEL_W_IN * PANEL_ASPECT),
                               constrained_layout=True)
        draw_family(ax, family, data)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(TITLE, pad=2)
        fig.savefig(os.path.join(out_dir, f"{family}_PSNR.pdf"), bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(TEXT_W_IN, 0.4))
    h, l = legend_handles(with_gt=True)
    fig.legend(h, l, loc="center", ncol=4, frameon=False, handletextpad=0.2,
               columnspacing=1.0)
    fig.savefig(os.path.join(out_dir, "legend.pdf"), bbox_inches="tight")
    plt.close(fig)


def option_c(data: dict, out_dir: str) -> None:
    """One 2x3 figure, shared y-axis, panel letters drawn in, common legend."""
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W_IN, TEXT_W_IN * 0.58), sharey=True,
                             layout="constrained")
    lo, hi = np.inf, -np.inf
    for k, (ax, family) in enumerate(zip(axes.flat, PAPER_ORDER)):
        a, b = draw_family(ax, family, data)
        lo, hi = min(lo, a), max(hi, b)
        ax.set_title(f"({'abcdef'[k]}) {FAMILY_NAME[family]}", fontsize=LABEL_FS, pad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel(TITLE)
    pad = 0.04 * (hi - lo)
    axes[0, 0].set_ylim(lo - pad, hi + pad)
    h, l = legend_handles(with_gt=True)
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False,
               handletextpad=0.2, columnspacing=1.0)
    fig.get_layout_engine().set(h_pad=0.06, hspace=0.08)
    fig.savefig(os.path.join(out_dir, "all_PSNR.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--options", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    args = p.parse_args()
    apply_style()
    data = load()
    for opt in args.options:
        out_dir = os.path.join(HERE, "print", f"opt{opt}")
        os.makedirs(out_dir, exist_ok=True)
        {"A": option_a, "B": option_b, "C": option_c}[opt](data, out_dir)
        print("wrote", out_dir, sorted(os.listdir(out_dir)))


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load and process motion blur CSVs with GT ---
def load_and_process(data_path, gt_path, dataset_type):
    df = pd.read_csv(data_path)
    df[['PSNR_mean','PSNR_std']] = df['PSNR'].apply(lambda x: pd.Series(parse_metric(x)))
    df['blur_num'] = df['blur_level'].str.extract(r'(\d+)').astype(int)
    df['dataset'] = dataset_type
    df['method_clean'] = df['method'].map({
        f"nerfacto_motion_blur_img": "Nerfacto",
        f"splatfacto_motion_blur_img": "Splatfacto"
    })
    df = df[['method_clean','dataset','blur_num','PSNR_mean','PSNR_std']]

    # Ground Truth
    df_gt = pd.read_csv(gt_path)
    df_gt = df_gt[df_gt['method'].isin([f"nerfacto_{dataset_type}", f"splatfacto_{dataset_type}"])].copy()
    df_gt['blur_num'] = 0
    df_gt['dataset'] = dataset_type
    df_gt['method_clean'] = df_gt['method'].map({
        f"nerfacto_{dataset_type}": "Nerfacto GT",
        f"splatfacto_{dataset_type}": "Splatfacto GT"
    })
    df_gt[['PSNR_mean','PSNR_std']] = df_gt['PSNR'].apply(lambda x: pd.Series(parse_metric(x)))
    df_gt = df_gt[['method_clean','dataset','blur_num','PSNR_mean','PSNR_std']]

    # Combine
    df_all = pd.concat([df_gt, df], ignore_index=True)
    return df_all

# --- Load indoor and outdoor ---
df_indoor = load_and_process(
    "/home/efe/data/metrics/motion_blur/motionblur_indoor_summary.csv",
    "/home/efe/data/metrics/original/original_indoor_summary.csv",
    "indoor"
)
df_outdoor = load_and_process(
    "/home/efe/data/metrics/motion_blur/motionblur_outdoor_summary.csv",
    "/home/efe/data/metrics/original/original_outdoor_summary.csv",
    "outdoor"
)
df_all = pd.concat([df_indoor, df_outdoor], ignore_index=True)

# --- Colors and markers ---
method_colors = {
    "Nerfacto": "#1f77b4", "Splatfacto": "#ff7f0e",
    "Nerfacto GT": "#0d3a66", "Splatfacto GT": "#b35600"
}
method_markers = {
    "Nerfacto": "o", "Splatfacto": "^",
    "Nerfacto GT": "o", "Splatfacto GT": "^"
}

# --- Offsets for overlapping markers ---
unique_combos = df_all['dataset'] + " " + df_all['method_clean']
offset_step = 0.2
combo_offsets = {combo: i*offset_step - offset_step*(len(unique_combos.unique())-1)/2
                 for i, combo in enumerate(unique_combos.unique())}

# --- Plot PSNR ---
fig, ax = plt.subplots(figsize=(8,6))
all_lows, all_highs = [], []

for combo in unique_combos.unique():
    df_sub = df_all[(df_all['dataset'] + " " + df_all['method_clean']) == combo]
    if df_sub.empty:
        continue

    x_vals = df_sub['blur_num'].values + combo_offsets[combo]
    method_name = df_sub['method_clean'].iloc[0]
    dataset_name = df_sub['dataset'].iloc[0]
    color = method_colors[method_name]
    marker = method_markers[method_name]
    y_mean = df_sub['PSNR_mean']
    y_std = df_sub['PSNR_std']

    all_lows.extend(y_mean - y_std)
    all_highs.extend(y_mean + y_std)

    # Determine marker facecolor
    if "GT" in method_name:
        # Hollow for Indoor GT, filled for Outdoor GT
        markerface = "none" if dataset_name=="indoor" else color
    else:
        # Filled for Outdoor non-GT, hollow for Indoor non-GT
        markerface = color if dataset_name=="outdoor" else "none"

    # Line for non-GT only
    if "GT" not in method_name:
        ax.plot(x_vals, y_mean, linestyle="-", color=color, alpha=0.2, zorder=1)

    # Error bars for non-GT only
    yerr_val = None if "GT" in method_name else y_std
    ax.errorbar(
        x_vals, y_mean, yerr=yerr_val, marker=marker, linestyle='none',
        capsize=3, markersize=10, color=color,
        markerfacecolor=markerface, markeredgewidth=1.5, zorder=2
    )

# --- Labels ---
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.set_xlabel("Motion Blur Level", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.set_xticks([0,10,20,30,40,50])
ax.set_xticklabels(["GT",10,20,30,40,50])

# --- Add Splatfacto GT Indoor value as left tick ---
splat_tick = 33.08
yticks = list(ax.get_yticks())
if splat_tick not in yticks:
    yticks.append(splat_tick)
yticks = sorted(yticks)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.2f}" for t in yticks])  # show numeric labels on left

# --- Secondary y-axis: mirror left ticks but hide labels ---
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(yticks)
ax2.set_yticklabels([])  # only tick marks
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)

# --- Custom legend like gamma plot ---
legend_order = [
    "Splatfacto GT Indoor", "Nerfacto GT Indoor",
    "Splatfacto GT Outdoor", "Nerfacto GT Outdoor",
    "Splatfacto Outdoor", "Nerfacto Outdoor",
    "Splatfacto Indoor", "Nerfacto Indoor"
]
legend_elements = []
for name in legend_order:
    parts = name.split(" ")
    method = parts[0]
    is_gt = "GT" in parts
    dataset = parts[-1].lower()  # indoor/outdoor
    color = method_colors[method + (" GT" if is_gt else "")]
    marker = method_markers[method + (" GT" if is_gt else "")]
    # Hollow for Indoor GT, filled for Outdoor GT; non-GT follows dataset color/fill
    if is_gt:
        facecolor = "none" if dataset=="indoor" else color
    else:
        facecolor = color if dataset=="outdoor" else "none"

    legend_elements.append(Line2D([0],[0], color=color, marker=marker, linestyle='none',
                                  markersize=10, markerfacecolor=facecolor, markeredgewidth=1.5))

ax.legend(legend_elements, legend_order, loc="best", fontsize=8)

fig.savefig("/home/efe/data/metrics/motion_blur/motionblur_PSNR_gamma_style_legend.png", dpi=300)
plt.show()

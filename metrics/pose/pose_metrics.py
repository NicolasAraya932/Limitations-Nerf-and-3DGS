import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load and process CSV ---
def load_and_process(data_path, gt_path=None, noise_levels=None, dataset="Indoor"):
    df = pd.read_csv(data_path)
    
    # Split metrics into mean/std
    for metric in ["PSNR"]:
        df[[f"{metric}_mean", f"{metric}_std"]] = df[metric].apply(lambda x: pd.Series(parse_metric(x)))
    
    # Assign numeric level for x-axis
    if noise_levels is not None:
        methods = df['method'].unique()
        all_rows = []
        for method in methods:
            df_method = df[df['method'] == method].copy()
            if len(df_method) != len(noise_levels):
                raise ValueError(f"Rows for {method} ({len(df_method)}) != noise_levels ({len(noise_levels)})")
            df_method['level_num'] = noise_levels
            df_method['dataset'] = dataset
            all_rows.append(df_method)
        df = pd.concat(all_rows, ignore_index=True)
    else:
        df['level_num'] = range(len(df))
        df['dataset'] = dataset

    # Add ground truth if provided
    if gt_path is not None:
        df_gt = pd.read_csv(gt_path)
        df_gt = df_gt[df_gt['method'].isin(['nerfacto_indoor','splatfacto_indoor','nerfacto_outdoor','splatfacto_outdoor'])]
        for metric in ["PSNR"]:
            df_gt[[f"{metric}_mean", f"{metric}_std"]] = df_gt[metric].apply(lambda x: pd.Series(parse_metric(x)))
        df_gt['method'] = df_gt['method'] + " Ground Truth"
        df_gt['scene'] = "GT"
        df_gt['level_num'] = 0.0
        df_gt['dataset'] = dataset
        df = pd.concat([df_gt, df], ignore_index=True)
    
    return df

# --- Paths ---
indoor_csv = "/home/efe/data/metrics/pose/pose_indoor_summary.csv"
outdoor_csv = "/home/efe/data/metrics/pose/pose_outdoor_summary.csv"
gt_indoor_csv = "/home/efe/data/metrics/original/original_indoor_summary.csv"
gt_outdoor_csv = "/home/efe/data/metrics/original/original_outdoor_summary.csv"

# --- Noise levels ---
noise_levels = [0.005, 0.01, 0.015, 0.02, 0.025]

# --- Load data ---
df_indoor = load_and_process(indoor_csv, gt_path=gt_indoor_csv, noise_levels=noise_levels, dataset="Indoor")
df_outdoor = load_and_process(outdoor_csv, gt_path=gt_outdoor_csv, noise_levels=noise_levels, dataset="Outdoor")

# --- Combine ---
df_all = pd.concat([df_indoor, df_outdoor], ignore_index=True)

# --- Clean method names ---
method_rename = {
    "nerfacto_indoor": "Nerfacto",
    "splatfacto_indoor": "Splatfacto",
    "nerfacto_outdoor": "Nerfacto",
    "splatfacto_outdoor": "Splatfacto",
    "nerfacto_indoor Ground Truth": "Nerfacto GT",
    "splatfacto_indoor Ground Truth": "Splatfacto GT",
    "nerfacto_outdoor Ground Truth": "Nerfacto GT",
    "splatfacto_outdoor Ground Truth": "Splatfacto GT"
}
df_all['method_clean'] = df_all['method'].replace(method_rename)

# --- Colors & markers ---
base_colors = {
    "Nerfacto": "#1f77b4",
    "Splatfacto": "#ff7f0e"
}
gt_colors = {
    "Nerfacto GT": "#0d3a66",   # darker blue
    "Splatfacto GT": "#b35600"  # darker orange
}
method_colors = {**base_colors, **gt_colors}

method_markers = {
    "Nerfacto": "o", "Splatfacto": "^",
    "Nerfacto GT": "o", "Splatfacto GT": "^"
}

# --- X-offsets to separate points like gamma plot ---
combos = df_all['dataset'] + " " + df_all['method_clean']
unique_combos = combos.unique()
offset_step = 0.0005  # small offset
combo_offsets = {combo: i*offset_step - offset_step*(len(unique_combos)-1)/2
                 for i, combo in enumerate(unique_combos)}

# --- Plot ---
fig, ax = plt.subplots(figsize=(8,6))
all_means, all_lows, all_highs = [], [], []

for combo in unique_combos:
    df_sub = df_all[(df_all['dataset'] + " " + df_all['method_clean']) == combo]
    if df_sub.empty:
        continue
    
    x_vals = df_sub['level_num'].values + combo_offsets[combo]
    y_mean = df_sub['PSNR_mean']
    y_std = df_sub['PSNR_std']
    
    all_means.extend(y_mean)
    all_lows.extend(y_mean - y_std)
    all_highs.extend(y_mean + y_std)
    
    dataset = df_sub['dataset'].iloc[0]
    method = df_sub['method_clean'].iloc[0]
    color = method_colors[method]
    marker = method_markers[method]
    facecolor = "none" if dataset=="Indoor" else color
    edgecolor = color
    
    # Faint line for non-GT
    if "GT" not in method:
        ax.plot(x_vals, y_mean, linestyle='-', color=color, alpha=0.2, zorder=1)
    
    # Plot markers + error bars (skip error bars for GT)
    yerr_val = None if "GT" in method else y_std
    ax.errorbar(x_vals, y_mean, yerr=yerr_val, marker=marker, linestyle='none',
                markersize=10, color=edgecolor, markerfacecolor=facecolor,
                markeredgewidth=1.5, capsize=3, zorder=2)

# Labels
ax.set_xlabel("Pose Noise", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.set_xticks([0] + noise_levels)
ax.set_xticklabels(["GT"] + noise_levels)
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.grid(False)

# --- Add fixed tick 33.08 on left axis ---
fixed_tick = 33.08
yticks = list(ax.get_yticks())
if not any(abs(fixed_tick - t) < 1e-6 for t in yticks):
    yticks.append(fixed_tick)
yticks = sorted(yticks)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.2f}" for t in yticks])  # show numeric labels on left

# --- Secondary y-axis: mirror left ticks but hide labels ---
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(yticks)
ax2.set_yticklabels([])  # tick marks only
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)

# --- Tick appearance: inside ---
ax.tick_params(axis='y', which='both', direction='in', left=True, right=False)
ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)

# --- Legend ---
legend_order = [
    "Splatfacto GT Indoor","Splatfacto GT Outdoor",
    "Nerfacto GT Indoor","Nerfacto GT Outdoor",
    "Splatfacto Indoor","Splatfacto Outdoor",
    "Nerfacto Indoor","Nerfacto Outdoor"
]
legend_elements = []
for name in legend_order:
    method_full, dataset = name.rsplit(" ",1)
    facecolor = "none" if "Indoor" in dataset else method_colors[method_full]
    handle = Line2D([0],[0], marker=method_markers[method_full],
                    color=method_colors[method_full], markerfacecolor=facecolor,
                    markersize=10, markeredgewidth=1.5, linestyle='None')
    legend_elements.append(handle)

ax.legend(legend_elements, legend_order, loc="best", fontsize=8)

# Save figure
#fig.savefig("/home/efe/data/metrics/pose/pose_all_PSNR_theme_shifted.png", dpi=300)
plt.show()

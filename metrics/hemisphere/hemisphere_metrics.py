import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load and process hemisphere CSVs with GT ---
def load_and_process(data_path, gt_path, dataset_type):
    df = pd.read_csv(data_path)
    df[['PSNR_mean', 'PSNR_std']] = df['PSNR'].apply(lambda x: pd.Series(parse_metric(x)))
    method_map = {
        f"nerfacto_{dataset_type}/hemisphere": "Nerfacto",
        f"splatfacto_{dataset_type}/hemisphere": "Splatfacto"
    }
    df['method_clean'] = df['method'].map(method_map)
    df['level_num'] = 1
    df['dataset'] = dataset_type
    df = df[['method_clean','dataset','level_num','PSNR_mean','PSNR_std']]

    # Load Ground Truth
    df_gt = pd.read_csv(gt_path)
    gt_map = {
        f"nerfacto_{dataset_type}": "Nerfacto GT",
        f"splatfacto_{dataset_type}": "Splatfacto GT"
    }
    df_gt = df_gt[df_gt['method'].isin(gt_map.keys())].copy()
    df_gt['method_clean'] = df_gt['method'].map(gt_map)
    df_gt['level_num'] = 0
    df_gt['dataset'] = dataset_type
    df_gt[['PSNR_mean','PSNR_std']] = df_gt['PSNR'].apply(lambda x: pd.Series(parse_metric(x)))
    df_gt = df_gt[['method_clean','dataset','level_num','PSNR_mean','PSNR_std']]

    # Combine
    df_all = pd.concat([df_gt, df], ignore_index=True)
    return df_all

# --- Load indoor and outdoor ---
df_indoor = load_and_process(
    "/home/efe/data/metrics/hemisphere/hemisphere_indoor_summary.csv",
    "/home/efe/data/metrics/original/original_indoor_summary.csv",
    "indoor"
)
df_outdoor = load_and_process(
    "/home/efe/data/metrics/hemisphere/hemisphere_outdoor_summary.csv",
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

# --- Offsets to avoid overlap ---
unique_combos = df_all['dataset'] + " " + df_all['method_clean']
offset_step = 0.005
combo_offsets = {combo: i*offset_step - offset_step*(len(unique_combos.unique())-1)/2
                 for i, combo in enumerate(unique_combos.unique())}

# --- Plot PSNR with distinct GT markers ---
fig, ax = plt.subplots(figsize=(8,6))
all_lows, all_highs = [], []

for combo in unique_combos.unique():
    df_sub = df_all[(df_all['dataset'] + " " + df_all['method_clean']) == combo]
    if df_sub.empty:
        continue

    x_vals = df_sub['level_num'].values + combo_offsets[combo]
    method_name = df_sub['method_clean'].iloc[0]
    dataset_name = df_sub['dataset'].iloc[0]
    color = method_colors[method_name]
    marker = method_markers[method_name]
    y_mean = df_sub["PSNR_mean"]
    y_std = df_sub["PSNR_std"]

    all_lows.extend(y_mean - y_std)
    all_highs.extend(y_mean + y_std)

    # Determine marker face color
    if "GT" in method_name:
        if dataset_name.lower() == "indoor":
            markerface = "none"  # hollow
        else:
            markerface = color   # filled
    else:
        markerface = "none" if dataset_name.lower() == "indoor" else color

    # Line for non-GT
    if "GT" not in method_name:
        ax.plot(x_vals, y_mean, linestyle='-', color=color, alpha=0.2, zorder=1)

    # Error bars / markers
    yerr_val = None if "GT" in method_name else y_std
    ax.errorbar(
        x_vals, y_mean, yerr=yerr_val, marker=marker, linestyle='none',
        capsize=3, markersize=10, color=color,
        markerfacecolor=markerface, markeredgewidth=1.5, zorder=2
    )


# --- Labels ---
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.set_xlabel("Hemisphere Setting", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.set_xticks([0,1])
ax.set_xticklabels(["GT","Hemisphere"])

# --- Add Splatfacto GT Indoor value as tick ---
splat_tick = 33.08
yticks = list(ax.get_yticks())
if splat_tick not in yticks:
    yticks.append(splat_tick)
yticks = sorted(yticks)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.2f}" for t in yticks])

# Mirror on secondary axis
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(yticks)
ax2.set_yticklabels([])  # only ticks


# --- Tick appearance: inside ---
ax.tick_params(axis='y', which='both', direction='in', left=True, right=False)
ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)

# --- Custom legend with matching GT markers ---
legend_order = [
    "Splatfacto GT Indoor",
    "Splatfacto GT Outdoor",
    "Nerfacto GT Indoor",
    "Nerfacto GT Outdoor",
    "Splatfacto Indoor",
    "Splatfacto Outdoor",
    "Nerfacto Indoor",
    "Nerfacto Outdoor"
]

legend_elements = []
for name in legend_order:
    method, dataset = name.rsplit(" ", 1)
    color = method_colors[method]
    marker = method_markers[method]

    # Match plot marker face color for GT and dataset
    if "GT" in method:
        if dataset.lower() == "indoor":
            facecolor = "none"  # hollow
        else:
            facecolor = color   # filled
    else:
        facecolor = "none" if dataset.lower() == "indoor" else color

    legend_elements.append(Line2D([0], [0], color=color, marker=marker,
                                  linestyle='none', markersize=10,
                                  markerfacecolor=facecolor, markeredgewidth=1.5))

ax.legend(legend_elements, legend_order, loc="upper center", fontsize=8)


#fig.savefig("/home/efe/data/metrics/hemisphere/hemisphere_PSNR_gamma_style.png", dpi=300)
plt.show()

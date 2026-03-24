import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load method CSV ---
def load_and_process(path, method_map, dataset_name):
    df = pd.read_csv(path)
    df[["PSNR_mean", "PSNR_std"]] = df["PSNR"].apply(lambda x: pd.Series(parse_metric(x)))
    df['level_num'] = df['setting'].str.extract(r'_(\d+)$')[0].astype(int)
    df['dataset'] = dataset_name
    df['method_clean'] = df['method'].map(method_map)
    return df

# --- Load GT ---
def load_gt(gt_csv, gt_methods, dataset_name):
    df_gt = pd.read_csv(gt_csv)
    df_gt = df_gt[df_gt['method'].isin(gt_methods.keys())].copy()
    df_gt["PSNR_mean"] = df_gt["PSNR"].apply(lambda x: parse_metric(x)[0])
    df_gt["PSNR_std"] = df_gt["PSNR"].apply(lambda x: parse_metric(x)[1])
    df_gt['level_num'] = 0
    df_gt['dataset'] = dataset_name
    df_gt['method_clean'] = df_gt['method'].map(gt_methods)
    return df_gt

# --- Paths and mapping ---
indoor_csv = "/home/efe/data/metrics/reduced_img/reduced_img_indoor_summary.csv"
outdoor_csv = "/home/efe/data/metrics/reduced_img/reduced_img_outdoor_summary.csv"
gt_indoor_csv = "/home/efe/data/metrics/original/original_indoor_summary.csv"
gt_outdoor_csv = "/home/efe/data/metrics/original/original_outdoor_summary.csv"

indoor_methods = {"nerfacto_indoor/reduced_img": "Nerfacto",
                  "splatfacto_indoor/reduced_img": "Splatfacto"}
outdoor_methods = {"nerfacto_outdoor/reduced_img": "Nerfacto",
                   "splatfacto_outdoor/reduced_img": "Splatfacto"}

gt_methods_indoor = {"nerfacto_indoor": "Nerfacto GT",
                     "splatfacto_indoor": "Splatfacto GT"}
gt_methods_outdoor = {"nerfacto_outdoor": "Nerfacto GT",
                      "splatfacto_outdoor": "Splatfacto GT"}

# --- Load data ---
df_indoor = load_and_process(indoor_csv, indoor_methods, "Indoor")
df_outdoor = load_and_process(outdoor_csv, outdoor_methods, "Outdoor")
df_gt_indoor = load_gt(gt_indoor_csv, gt_methods_indoor, "Indoor")
df_gt_outdoor = load_gt(gt_outdoor_csv, gt_methods_outdoor, "Outdoor")

df_all = pd.concat([df_indoor, df_outdoor, df_gt_indoor, df_gt_outdoor], ignore_index=True)

# --- Assign positions ---
levels_sorted = sorted(df_all['level_num'].unique())
level_to_pos = {lvl: i for i, lvl in enumerate(levels_sorted)}
df_all['plot_pos'] = df_all['level_num'].map(level_to_pos)

# Minor shifts to avoid overlapping
shift_val = 0.1
df_all['plot_pos_shifted'] = df_all['plot_pos']
df_all.loc[df_all['method_clean'].str.contains("Splatfacto"), 'plot_pos_shifted'] += shift_val
df_all.loc[df_all['method_clean'].str.contains("Nerfacto"), 'plot_pos_shifted'] -= shift_val / 2

# --- Colors & markers ---
method_colors = {
    "Nerfacto": "#1f77b4",
    "Splatfacto": "#ff7f0e",
    "Nerfacto GT": "#0d3a66",
    "Splatfacto GT": "#b35600"
}

method_markers = {
    "Nerfacto": "o",
    "Splatfacto": "^",
    "Nerfacto GT": "o",
    "Splatfacto GT": "^"
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(8,6))
all_lows, all_highs = [], []

unique_combos = df_all['dataset'] + " " + df_all['method_clean']
combo_offsets = {combo: i*0.2 - 0.2*(len(unique_combos.unique())-1)/2 for i, combo in enumerate(unique_combos.unique())}

for combo in unique_combos.unique():
    df_sub = df_all[(df_all['dataset'] + " " + df_all['method_clean']) == combo]
    if df_sub.empty: 
        continue

    x_vals = df_sub['plot_pos_shifted'].values
    y_mean = df_sub['PSNR_mean'].values
    y_std = df_sub['PSNR_std'].values
    method_name = df_sub['method_clean'].iloc[0]
    color = method_colors[method_name]
    marker = method_markers[method_name]

    # Determine marker facecolor (Indoor hollow, Outdoor filled)
    facecolor = 'none' if df_sub['dataset'].iloc[0] == "Indoor" else color

    if "GT" in method_name:
        ax.errorbar(x_vals, y_mean, yerr=y_std, marker=marker, linestyle='none',
                    markersize=10, color=color, markerfacecolor=facecolor, markeredgewidth=1.5, zorder=2)
    else:
        # Semi-transparent line connecting the points
        ax.plot(x_vals, y_mean, color=color, alpha=0.2, zorder=1)
        # Error bars with proper marker facecolor
        ax.errorbar(x_vals, y_mean, yerr=y_std, marker=marker, linestyle='none',
                    capsize=3, markersize=8, color=color,
                    markerfacecolor=facecolor, markeredgewidth=1.5, zorder=2)


    all_lows.extend(y_mean - y_std)
    all_highs.extend(y_mean + y_std)

# --- Labels ---
ax.set_xticks(list(level_to_pos.values()))
ax.set_xticklabels(["GT" if x==0 else f"{x}th image" for x in levels_sorted])
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.set_xlabel("Subimages", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.grid(False)

# --- Add fixed tick 33.08 on left axis ---
fixed_tick = 33.08
yticks = list(ax.get_yticks())
if not any(abs(fixed_tick - t) < 1e-6 for t in yticks):
    yticks.append(fixed_tick)
yticks = sorted(yticks)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.2f}" for t in yticks])  # show numeric labels on left



# Main axis ticks: only left + bottom
ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)


# --- Legend ---
legend_order = [
    "Splatfacto GT Indoor",
    "Splatfacto GT Outdoor",
    "Nerfacto GT Indoor",
    "Nerfacto GT Outdoor",
    "Splatfacto Indoor", "Splatfacto Outdoor",
    "Nerfacto Indoor", "Nerfacto Outdoor"
]

legend_elements = []
for name in legend_order:
    method_full, dataset = name.rsplit(" ", 1)
    method = method_full
    color = method_colors[method]
    marker = method_markers[method]

    # Indoor hollow, Outdoor filled
    facecolor = "none" if dataset == "Indoor" else color

    handle = Line2D([0], [0], marker=marker, color=color, linestyle='None',
                    markersize=8, markerfacecolor=facecolor, markeredgewidth=1.2)
    legend_elements.append(handle)

ax.legend(legend_elements, legend_order, loc="best", fontsize=8, markerscale=0.8)

# --- Save + show ---
output_path = "/home/efe/data/metrics/reduced_img/reduced_all_PSNR_gamma_style_connected.png"
#fig.savefig(output_path, dpi=300)
plt.show()

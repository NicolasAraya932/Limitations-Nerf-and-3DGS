import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load and process CSV ---
def load_and_process(path, dataset, gt_path=None):
    df = pd.read_csv(path)
    for metric in ["PSNR", "SSIM", "LPIPS"]:
        df[[f"{metric}_mean", f"{metric}_std"]] = df[metric].apply(lambda x: pd.Series(parse_metric(x)))
    df['level_num'] = df['level'].str.extract(r'(\d+)').astype(int)
    df['dataset'] = dataset
    
    # Add GT if provided
    if gt_path is not None:
        df_gt = pd.read_csv(gt_path)
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            df_gt[[f"{metric}_mean", f"{metric}_std"]] = df_gt[metric].apply(lambda x: pd.Series(parse_metric(x)))
        df_gt['method'] = df_gt['method'] + " GT"
        df_gt['level'] = "level_0"
        df_gt['level_num'] = 0
        df_gt['dataset'] = dataset
        df = pd.concat([df, df_gt], ignore_index=True)
    
    return df

# --- Normalize levels (map to 0..N) ---
def normalize_levels(df):
    unique_levels = sorted(df['level_num'].unique())
    mapping = {lvl: i for i, lvl in enumerate(unique_levels)}
    df['level_num'] = df['level_num'].map(mapping)
    return df

# --- Paths ---
indoor_csv = "/home/efe/data/metrics/height/height_indoor_summary.csv"
indoor_gt  = "/home/efe/data/metrics/original/original_indoor_summary.csv"
outdoor_csv = "/home/efe/data/metrics/height/height_outdoor_summary.csv"
outdoor_gt  = "/home/efe/data/metrics/original/original_outdoor_summary.csv"

# --- Load + normalize ---
df_indoor = load_and_process(indoor_csv, "Indoor", gt_path=indoor_gt)
df_indoor = normalize_levels(df_indoor)

df_outdoor = load_and_process(outdoor_csv, "Outdoor", gt_path=outdoor_gt)  # include GT now
df_outdoor = normalize_levels(df_outdoor)

# Shift outdoor levels by +1 for non-GT
df_outdoor.loc[df_outdoor['method'].str.contains("GT") == False, 'level_num'] += 1

df_height = pd.concat([df_indoor, df_outdoor], ignore_index=True)

# --- Clean method names ---
method_rename = {
    "nerfacto_indoor/height": "Nerfacto",
    "splatfacto_indoor/height": "Splatfacto",
    "nerfacto_outdoor/height": "Nerfacto",
    "splatfacto_outdoor/height": "Splatfacto",
    "nerfacto_indoor GT": "Nerfacto GT",
    "splatfacto_indoor GT": "Splatfacto GT",
    "nerfacto_outdoor GT": "Nerfacto GT",
    "splatfacto_outdoor GT": "Splatfacto GT",
}
df_height['method_clean'] = df_height['method'].replace(method_rename)
df_height = df_height.dropna(subset=['method_clean'])
df_height = df_height.sort_values(["dataset", "level_num"])

# --- Colors ---
method_colors = {
    "Nerfacto": "#1f77b4",       # Blue
    "Splatfacto": "#ff7f0e",     # Orange
    "Nerfacto GT": "#0d3a66",    # Darker Blue
    "Splatfacto GT": "#b35600",  # Darker Orange
}

# --- Markers ---
method_markers = {
    "Nerfacto": "o",
    "Splatfacto": "^",
    "Nerfacto GT": "o",
    "Splatfacto GT": "^"
}

# --- Plot PSNR only ---
fig, ax = plt.subplots(figsize=(8, 6))

all_lows, all_highs = [], []

unique_combos = df_height['dataset'] + " " + df_height['method_clean']
offset_step = 0.05
combo_offsets = {combo: i*offset_step - offset_step*(len(unique_combos.unique())-1)/2
                 for i, combo in enumerate(unique_combos.unique())}

for combo in unique_combos.unique():
    dataset, method = combo.split(" ", 1)
    df_sub = df_height[(df_height['dataset'] == dataset) & (df_height['method_clean'] == method)]
    if df_sub.empty:
        continue
    
    x_vals = df_sub['level_num'].values + combo_offsets[combo]
    color = method_colors[method]
    marker = method_markers[method]
    facecolor = "none" if dataset == "Indoor" else color
    
    if "GT" in method:
        ax.errorbar(
            x_vals,
            df_sub["PSNR_mean"],
            yerr=None,
            marker=marker,
            linestyle="none",
            color=color,
            markersize=10,
            markerfacecolor=facecolor,
            markeredgewidth=1.5,
            zorder=2
        )
    else:
        ax.plot(
            x_vals,
            df_sub["PSNR_mean"],
            linestyle="-",
            color=color,
            alpha=0.2,
            zorder=1
        )
        ax.errorbar(
            x_vals,
            df_sub["PSNR_mean"],
            yerr=df_sub["PSNR_std"],
            marker=marker,
            linestyle="none",
            capsize=2,
            color=color,
            markersize=10,
            markerfacecolor=facecolor,
            markeredgewidth=1.5,
            zorder=2
        )
    
    all_lows.extend(df_sub["PSNR_mean"] - df_sub["PSNR_std"])
    all_highs.extend(df_sub["PSNR_mean"] + df_sub["PSNR_std"])

# Labels
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.set_xlabel("Height Level", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.set_xticks(sorted(df_height['level_num'].unique()))
ax.set_xticklabels(["GT" if x==0 else x for x in sorted(df_height['level_num'].unique())])

# ✅ Existing left ticks
left_ticks = list(ax.get_yticks())

# ✅ Compute min + max from right-value space
right_min = np.min(all_lows)
right_max = np.max(all_highs)

# ✅ Add min + max from right → left axis
# (avoid duplicates)
extra_ticks = []
#if right_min not in left_ticks:
    #extra_ticks.append(right_min)
if right_max not in left_ticks:
    extra_ticks.append(right_max)

ax.set_yticks(left_ticks + extra_ticks)

# ✅ Format left labels same + added min/max
ax.set_yticklabels([f"{t:.2f}" for t in ax.get_yticks()])

# ✅ Create right axis
ax2 = ax.twinx()
ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin, ymax)

# ✅ Hide right tick labels but keep the axis
ax2.set_yticks([])   # no labels
ax2.set_ylabel("")   # no name

# Tick appearance
ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)





# Legend
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
    method, dataset = name.rsplit(" ", 1)
    color = method_colors[method]
    marker = method_markers[method]
    facecolor = "none" if dataset == "Indoor" else color
    handle = Line2D([0], [0], marker=marker, color=color, linestyle='None',
                    markersize=10, markerfacecolor=facecolor, markeredgewidth=1.5)
    legend_elements.append(handle)

ax.legend(legend_elements, legend_order, loc="best", fontsize=8, markerscale=0.8)

#output_path = "/home/efe/data/metrics/height/height_PSNR_with_outdoorGT.png"
#fig.savefig(output_path, dpi=300)
plt.show()

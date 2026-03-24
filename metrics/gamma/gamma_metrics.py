import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

def load_and_process(path, dataset, gamma_value=None):
    df = pd.read_csv(path)
    for metric in ["PSNR", "SSIM", "LPIPS"]:
        df[[f"{metric}_mean", f"{metric}_std"]] = df[metric].apply(lambda x: pd.Series(parse_metric(x)))
    if gamma_value is not None:
        df["gamma"] = gamma_value
    else:
        df["gamma"] = df['setting'].str.extract(r'perc_(\d+)').astype(int)
    df["dataset"] = dataset
    return df

# --- Load datasets ---
indoor_gamma = load_and_process("/home/efe/data/metrics/gamma/gamma_indoor_summary.csv", "Indoor")
indoor_original = load_and_process("/home/efe/data/metrics/original/original_indoor_summary.csv", "Indoor", gamma_value=0)
outdoor_gamma = load_and_process("/home/efe/data/metrics/gamma/gamma_outdoor_summary.csv", "Outdoor")
outdoor_original = load_and_process("/home/efe/data/metrics/original/original_outdoor_summary.csv", "Outdoor", gamma_value=0)

df = pd.concat([indoor_gamma, indoor_original, outdoor_gamma, outdoor_original], ignore_index=True)

# --- Clean method names ---
method_rename = {
    "nerfacto_indoor/gamma": "Nerfacto",
    "splatfacto_indoor/gamma": "Splatfacto",
    "nerfacto_outdoor/gamma": "Nerfacto",
    "splatfacto_outdoor/gamma": "Splatfacto",
    "nerfacto_indoor": "Nerfacto GT",
    "splatfacto_indoor": "Splatfacto GT",
    "nerfacto_outdoor": "Nerfacto GT",
    "splatfacto_outdoor": "Splatfacto GT",
}
df['method_clean'] = df['method'].map(method_rename)
df = df.dropna(subset=['method_clean'])
df = df.sort_values("gamma")

# --- Colors per method ---
method_colors = {
    "Nerfacto": "#1f77b4",       # Blue
    "Splatfacto": "#ff7f0e",     # Orange
    "Nerfacto GT": "#0d3a66",    # Darker Blue
    "Splatfacto GT": "#b35600",  # Darker Orange
}

# --- Marker shapes per method ---
method_markers = {
    "Nerfacto": "o",        # Circle
    "Splatfacto": "^",      # Triangle up
    "Nerfacto GT": "o",
    "Splatfacto GT": "^"
}

# --- Line style by dataset ---
dataset_styles = {
    "Indoor": "-",
    "Outdoor": "-"
}

# --- Offset values for each line (slight separation) ---
unique_combos = df['dataset'] + " " + df['method_clean']
# --- Offset per method per dataset ---
offset_val = 0.5  # how much to separate lines from the center

# Explicit offsets (relative to gamma)
combo_offsets = {
    "Outdoor Splatfacto": -1.0,  # leftmost
    "Outdoor Nerfacto": -0.5,
    "Indoor Splatfacto": 1.0,    # rightmost
    "Indoor Nerfacto": 0.5
    # GT methods can have same offsets if desired
}

# --- Plot PSNR only ---
fig, ax = plt.subplots(figsize=(8, 6))
gamma_unique = sorted(df['gamma'].unique())

# Track global min/max for scaling
all_means = []
all_lows = []
all_highs = []

for combo in unique_combos.unique():
    df_sub = df[(df['dataset'] + " " + df['method_clean']) == combo]
    if df_sub.empty:
        continue

    combo_key = f"{df_sub['dataset'].iloc[0]} {df_sub['method_clean'].iloc[0]}"
    x_vals = df_sub['gamma'].values + combo_offsets.get(combo_key, 0)

    # Color, line style, marker
    color = method_colors[df_sub['method_clean'].iloc[0]]
    linestyle = dataset_styles[df_sub['dataset'].iloc[0]]
    marker = method_markers[df_sub['method_clean'].iloc[0]]

    # PSNR mean/std
    y_mean = df_sub["PSNR_mean"]
    y_std = df_sub["PSNR_std"]

    # Save ranges
    all_means.extend(y_mean)
    all_lows.extend(y_mean - y_std)
    all_highs.extend(y_mean + y_std)

    # Decide hollow vs filled (Indoor hollow, Outdoor filled)
    if df_sub['dataset'].iloc[0] == "Indoor":
        markerface = "none"
    else:
        markerface = color

    # Plot connecting line with low opacity (skip GT)
    if "GT" not in df_sub['method_clean'].iloc[0]:
        ax.plot(
            x_vals,
            y_mean,
            linestyle=linestyle,
            color=color,
            alpha=0.2,
            zorder=1
        )

    # Plot markers with error bars (skip error bars for GT)
    if df_sub['method_clean'].iloc[0] in ["Nerfacto GT", "Splatfacto GT"]:
        yerr_val = None
    else:
        yerr_val = y_std

    ax.errorbar(
        x_vals,
        y_mean,
        yerr=yerr_val,
        marker=marker,
        linestyle='none',
        capsize=2,
        markersize=10,
        color=color,
        markerfacecolor=markerface,
        markeredgewidth=1.5,
        zorder=2
    )

# Labels
ax.set_title("PSNR (dB) ↑", fontsize=16)
ax.set_xlabel("Percentage of images with gamma noise", fontsize=16)
ax.set_ylabel("PSNR (dB)")
ax.set_xticks(gamma_unique)
ax.set_xticklabels(["GT" if x == 0 else x for x in gamma_unique])

# Secondary axis for right (already exists)
ax2 = ax.twinx()
ax2.set_ylabel("")  # keep empty if you want
# Right axis values min/max
right_min = min(all_lows)
right_max = max(all_highs)

# Left axis: get current ticks
left_ticks = ax.get_yticks().tolist()

# Add right axis min/max if not already in left ticks
combined_ticks = sorted(set(left_ticks + [right_min, right_max]))

# Update left axis ticks and labels
ax.set_yticks(combined_ticks)
ax.set_yticklabels([f"{t:.2f}" for t in combined_ticks])

# Remove right axis labels but keep spine
ax2.set_yticks([])  # no tick labels

# --- Custom legend in desired order ---
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

# Main axis ticks: only left + bottom
ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)

# Secondary axis ticks: only right
ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)

#plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save figure
#output_path = "/home/efe/data/metrics/gamma/gamma_PSNR_with_std_ticks_NEW.png"
#fig.savefig(output_path, dpi=300)

plt.show()

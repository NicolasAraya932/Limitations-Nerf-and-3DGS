import pandas as pd
import matplotlib.pyplot as plt

# --- Helper to parse mean ± std ---
def parse_metric(s):
    mean, std = s.split(" ± ")
    return float(mean), float(std)

# --- Load and process CSV ---
def load_and_process(data_path, gt_path=None):
    df = pd.read_csv(data_path)
    
    # Split metrics into mean/std
    for metric in ["PSNR", "SSIM", "LPIPS"]:
        df[[f"{metric}_mean", f"{metric}_std"]] = df[metric].apply(lambda x: pd.Series(parse_metric(x)))
    
    # Extract rotation value as float from scene name
    df['level_num'] = df['scene'].str.extract(r'rot([0-9.]+)_trans')[0].astype(float)
    
    # Add ground truth if provided
    if gt_path is not None:
        df_gt = pd.read_csv(gt_path)
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            df_gt[[f"{metric}_mean", f"{metric}_std"]] = df_gt[metric].apply(lambda x: pd.Series(parse_metric(x)))
        df_gt['method'] = df_gt['method'] + " Ground Truth"
        df_gt['scene'] = "GT"
        df_gt['level_num'] = 0.0
        df = pd.concat([df_gt, df], ignore_index=True)
    
    return df

# --- Paths ---
data_csv = "/home/efe/data/metrics/pose/pose_outdoor_summary.csv"
gt_csv = "/home/efe/data/metrics/original/original_outdoor_summary.csv"  # Original clean outdoor CSV

# --- Load data ---
df_pose = load_and_process(data_csv, gt_path=gt_csv)

# --- Clean method names ---
method_rename = {
    "nerfacto_outdoor": "Nerfacto",
    "splatfacto_outdoor": "Splatfacto",
    "nerfacto_outdoor Ground Truth": "Nerfacto Ground Truth",
    "splatfacto_outdoor Ground Truth": "Splatfacto Ground Truth"
}
df_pose['method_clean'] = df_pose['method'].replace(method_rename)

# --- Base methods and colors ---
base_methods = ["Nerfacto", "Splatfacto"]
colors = plt.cm.tab10.colors
method_colors = {m: colors[i] for i, m in enumerate(base_methods)}
method_colors["Nerfacto Ground Truth"] = method_colors["Nerfacto"]
method_colors["Splatfacto Ground Truth"] = method_colors["Splatfacto"]

# --- Metrics and titles ---
metrics = ["PSNR", "SSIM", "LPIPS"]
titles = {"PSNR": "PSNR ↑", "SSIM": "SSIM ↑", "LPIPS": "LPIPS ↓"}

# --- Sort for plotting ---
df_pose = df_pose.sort_values(['method_clean', 'level_num'])

# --- Create figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

for ax, metric in zip(axes, metrics):
    for method in df_pose['method_clean'].unique():
        df_method = df_pose[df_pose['method_clean'] == method]

        if "Ground Truth" in method:
            # Star marker for Ground Truth
            ax.errorbar(
                df_method['level_num'],
                df_method[f"{metric}_mean"],
                yerr=df_method[f"{metric}_std"],
                marker="*",
                linestyle="none",
                color=method_colors[method],
                markersize=12
            )
        else:
            # Circle + line for rotation/noise
            ax.errorbar(
                df_method['level_num'],
                df_method[f"{metric}_mean"],
                yerr=df_method[f"{metric}_std"],
                marker="o",
                linestyle="-",
                capsize=5,
                color=method_colors[method],
                markersize=6
            )

    ax.set_title(titles[metric], fontsize=14)
    ax.set_xlabel("Pose Noise")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(df_pose['level_num'].unique()))
    ax.set_xticklabels(["GT" if x==0 else x for x in sorted(df_pose['level_num'].unique())])
    ax.grid(False)

# --- Global legend ---
handles, labels = [], []
for method in df_pose['method_clean'].unique():
    handles.append(plt.Line2D([0], [0],
                              color=method_colors[method],
                              marker="*" if "Ground Truth" in method else "o",
                              linestyle="-" if "Ground Truth" not in method else "none",
                              markersize=10))
    labels.append(method)
fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=12)

# --- Caption below the plots ---
fig.text(0.5, 0.01, "Pose Outdoor Set", ha='center', fontsize=16)

# --- Adjust layout ---
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# --- Save figure ---
output_path = "/home/efe/data/metrics/pose/pose_outdoor_plot.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

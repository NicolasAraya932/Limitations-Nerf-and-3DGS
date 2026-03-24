import wandb
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
ENTITY = "eferdal-leibniz-universit-t-hannover"
PROJECT = "nerfstudio-project"
METHOD_PREFIXES = ["splatfacto_indoor/original_rep", "nerfacto_indoor/original_seed"]
METRICS = [
    "Eval Images Metrics Dict (all images)/psnr",
    "Eval Images Metrics Dict (all images)/ssim",
    "Eval Images Metrics Dict (all images)/lpips"
]

# --- Connect to W&B API ---
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# --- Helper to convert seconds to h:m:s ---
def format_duration(seconds):
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

# --- Collect relevant runs ---
data = []
for run in tqdm(runs, desc="Filtering runs"):
    name = run.name.strip("/")
    if any(name.startswith(prefix) for prefix in METHOD_PREFIXES):
        metrics = {metric: run.summary.get(metric, None) for metric in METRICS}
        metrics['Train Time Sec'] = run.summary.get("Train Total (time)", None)
        metrics['method'] = name.split("/")[0]  # splatfacto_indoor or nerfacto_indoor
        data.append(metrics)

if not data:
    print("❌ No matching runs found.")
    exit()

df = pd.DataFrame(data)
df[METRICS] = df[METRICS].astype(float)

# --- Group by method and compute mean/std ---
summary = df.groupby('method').agg(
    psnr_mean=(METRICS[0], "mean"),
    psnr_std=(METRICS[0], "std"),
    ssim_mean=(METRICS[1], "mean"),
    ssim_std=(METRICS[1], "std"),
    lpips_mean=(METRICS[2], "mean"),
    lpips_std=(METRICS[2], "std"),
    mean_train_time_sec=("Train Time Sec", "mean")
).reset_index()

# Convert mean training time back to h:m:s
summary['Mean Train Time'] = summary['mean_train_time_sec'].apply(lambda x: format_duration(x))
summary = summary.drop(columns=['mean_train_time_sec'])

# Format metrics as "mean ± std"
summary['PSNR'] = summary.apply(lambda r: f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}", axis=1)
summary['SSIM'] = summary.apply(lambda r: f"{r['ssim_mean']:.3f} ± {r['ssim_std']:.3f}", axis=1)
summary['LPIPS'] = summary.apply(lambda r: f"{r['lpips_mean']:.3f} ± {r['lpips_std']:.3f}", axis=1)

# Keep only formatted columns + method + train time
summary = summary[['method', 'Mean Train Time', 'PSNR', 'SSIM', 'LPIPS']]

# --- Save CSV ---
summary.to_csv("original_indoor_summary.csv", index=False)
print(f"✅ Saved summarized metrics to original_indoor_summary.csv")

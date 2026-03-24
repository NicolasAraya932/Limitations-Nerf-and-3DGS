import numpy as np
import cv2 as cv
import os
import shutil

# -------------------- Configuration --------------------
base_folder = "/media/efe/EFE/Datasets_AILAB/indoor/original"  # Original dataset images
depth_folder = "/media/efe/EFE/Datasets_AILAB/indoor/original/depth_maps"  # Precomputed depth maps
output_root = "/media/efe/EFE/Datasets_AILAB/motion_blur_img/indoor_depth"  # Root output folder

image_folder_name = "images_4"  # Folder with images
blur_length = 19  # Strong kernel size
blur_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # Percent of images to apply blur
num_seeds = 10  # Number of random seeds
depth_levels = 10  # Number of depth levels for depth-aware blur

required_files = ["transforms.json", "metadata.json"]  # Keep these files in output

# -------------------- Helper Functions --------------------
def selective_copy(source_path, output_path):
    """Copy required files and image folder."""
    os.makedirs(output_path, exist_ok=True)

    # Copy required files
    for file in required_files:
        s_file = os.path.join(source_path, file)
        if os.path.exists(s_file):
            shutil.copy2(s_file, output_path)

    # Copy image folder
    s_img_folder = os.path.join(source_path, image_folder_name)
    d_img_folder = os.path.join(output_path, image_folder_name)
    if os.path.exists(s_img_folder):
        shutil.copytree(s_img_folder, d_img_folder, dirs_exist_ok=True)


def motion_blur_kernel(size, angle_rad):
    """Generates a motion blur kernel in the direction of angle (in radians)."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    sin_a, cos_a = np.sin(angle_rad), np.cos(angle_rad)
    for i in range(size):
        x = int(center + (i - center) * cos_a)
        y = int(center + (i - center) * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel /= np.sum(kernel)
    return kernel

# -------------------- Main Script --------------------
if __name__ == "__main__":
    for seed in range(num_seeds):
        np.random.seed(seed)
        seed_folder = os.path.join(output_root, f"seed_{seed}")
        os.makedirs(seed_folder, exist_ok=True)

        for blur_rate in blur_rates:
            rate_percent = int(blur_rate * 100)
            output_path = os.path.join(seed_folder, f"outdoor_blur_{rate_percent}")
            img_folder_path = os.path.join(output_path, image_folder_name)

            # Copy images and required files if not already copied
            if not os.path.exists(output_path):
                print(f"[Seed {seed}] Copying data to {output_path}")
                selective_copy(base_folder, output_path)
            else:
                print(f"[Seed {seed}] Folder {output_path} exists. Skipping copy.")

            img_files = sorted([f for f in os.listdir(img_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            num_images = len(img_files) - 1  # last image has no successor

            num_blur_images = int(num_images * blur_rate)
            selected_indices = np.random.choice(num_images, num_blur_images, replace=False)
            selected_indices_set = set(selected_indices)

            print(f"[Seed {seed}] Applying depth-aware blur to {num_blur_images} images (rate {blur_rate})")

            for idx in range(num_images):
                if idx not in selected_indices_set:
                    continue

                img_path_1 = os.path.join(img_folder_path, img_files[idx])
                img_path_2 = os.path.join(img_folder_path, img_files[idx + 1])
                depth_path = os.path.join(depth_folder, img_files[idx].replace(".JPG", ".npy").replace(".jpg", ".npy"))

                img1 = cv.imread(img_path_1)
                img2 = cv.imread(img_path_2)
                if img1 is None or img2 is None:
                    print(f"[Seed {seed}] Failed to load {img_path_1} or {img_path_2}")
                    continue

                # Load precomputed depth map
                if not os.path.exists(depth_path):
                    print(f"[Seed {seed}] Depth map not found for {img_files[idx]}, skipping")
                    continue
                depth_map = np.load(depth_path)
                depth_norm = depth_map / (depth_map.max() + 1e-6)  # Normalize depth 0-1

                # Compute optical flow
                gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
                gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                dx = np.mean(flow[..., 0])
                dy = np.mean(flow[..., 1])
                angle = np.arctan2(dy, dx)

                # Depth-aware blur (nearer objects smear more)
                blurred_img = np.zeros_like(img1)
                levels = depth_levels
                depth_bins = np.linspace(0, 1, levels + 1)
                depth_labels = np.digitize(depth_norm, depth_bins)

                for l in range(1, levels + 1):
                    mask = (depth_labels == l)
                    # reverse the kernel size: nearer objects (low depth_norm) -> larger blur
                    kernel_size = max(1, int(blur_length * (levels - l + 1) / levels))
                    kernel = motion_blur_kernel(kernel_size, angle)
                    blurred_layer = cv.filter2D(img1, -1, kernel, borderType=cv.BORDER_REPLICATE)
                    blurred_img[mask] = blurred_layer[mask]

                cv.imwrite(img_path_1, blurred_img)
                print(f"[Seed {seed}] Blurred {img_files[idx]} with angle {np.degrees(angle):.2f}°")
    
    print("All depth-aware motion blur processing complete.")

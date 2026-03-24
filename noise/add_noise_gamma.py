import numpy as np
import cv2 as cv
import os
import shutil

# Configuration
base_folder_name = "garden_manhattan_pro"  # Original dataset folder
base_output_root = "/media/efe/EFE/Datasets_AILAB/outdoor/gamma"  # Root output path
output_prefix = "garden_pro_gamma"  # Prefix for generated dataset folders
gamma_values = (0.2, 5.0)  # Two gamma levels
required_files = ["sparse_pc.ply", "transforms.json"]  # Files to keep
image_folders = ["images_4"]  # Image folders to include

# Gamma noise function
def add_gamma_noise(image, gamma):
    image = image / 255.0
    image = np.power(image, gamma)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image

# Selective copy of required files and image folders
def selective_copy(source_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Copy required files
    for file in required_files:
        s_file = os.path.join(source_path, file)
        if os.path.exists(s_file):
            shutil.copy2(s_file, output_path)

    # Copy image folders
    for folder in image_folders:
        s_folder = os.path.join(source_path, folder)
        d_folder = os.path.join(output_path, folder)
        if os.path.exists(s_folder):
            shutil.copytree(s_folder, d_folder)

# Run gamma noise generation
def run_for_seed_and_perc(seed, noise_image_perc):
    np.random.seed(seed)
    perc_tag = f"{int(noise_image_perc * 100)}"

    # Create seed folder inside base_output_root
    seed_folder = os.path.join(base_output_root, f"seed_{seed}")
    os.makedirs(seed_folder, exist_ok=True)

    output_folder_name = f"{output_prefix}_perc_{perc_tag}"
    output_path = os.path.join(seed_folder, output_folder_name)

    source_path = os.path.join(os.getcwd(), base_folder_name)

    # Selectively copy only needed files and folders
    if not os.path.exists(output_path):
        print(f"[Seed {seed} | {perc_tag}%] Copying to {output_path}")
        selective_copy(source_path, output_path)
    else:
        print(f"[Seed {seed} | {perc_tag}%] Output folder exists, skipping copy.")

    # Process images in each folder
    for folder in image_folders:
        image_folder_path = os.path.join(output_path, folder)
        if not os.path.exists(image_folder_path):
            print(f"[{folder}] does not exist in {output_path}, skipping.")
            continue

        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        subset_size = int(len(image_files) * noise_image_perc)
        selected_filenames = np.random.choice(image_files, subset_size, replace=False)

        half = subset_size // 2
        gamma_02_files = set(selected_filenames[:half])
        gamma_50_files = set(selected_filenames[half:])

        for file_name in image_files:
            if file_name not in selected_filenames:
                continue  # skip non-selected

            img_path = os.path.join(image_folder_path, file_name)
            img = cv.imread(img_path)
            if img is None:
                print(f"[Seed {seed}] Failed to load {img_path}")
                continue

            if file_name in gamma_02_files:
                noisy_img = add_gamma_noise(img, gamma_values[0])
            elif file_name in gamma_50_files:
                noisy_img = add_gamma_noise(img, gamma_values[1])
            else:
                continue

            cv.imwrite(img_path, noisy_img)

# Main
if __name__ == "__main__":
    for perc in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for seed in range(10):
            run_for_seed_and_perc(seed, perc)

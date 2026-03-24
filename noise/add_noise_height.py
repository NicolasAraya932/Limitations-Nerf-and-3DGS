import os
import shutil
import json
import numpy as np

# -------------------- Parameters to Change --------------------
folder_name = "bonzai_manhattan_pro"       # Source dataset folder
base_output_root = "/media/efe/EFE/Datasets_AILAB/indoor/height"
image_folders_to_filter = ["images_4"]
max_images_per_level = 33  # Limit to 30 images max per elevation level


required_files = ["sparse_pc.ply", "transforms.json"]
image_folders = ["images_4"]  # Only copy this folder



# --------------------------------------------------------------

def selective_copy(source_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for item in os.listdir(source_path):
        s_item_path = os.path.join(source_path, item)
        d_item_path = os.path.join(output_path, item)
        if os.path.isdir(s_item_path):
            if item in image_folders:
                print(f"Copying folder: {item}")
                shutil.copytree(s_item_path, d_item_path)
        elif item in required_files:
            print(f"Copying file: {item}")
            shutil.copy2(s_item_path, d_item_path)

source_path = os.path.join(os.getcwd(), folder_name)
source_transforms_path = os.path.join(source_path, "transforms.json")

with open(source_transforms_path, "r") as f:
    original_transforms = json.load(f)

z_values = [np.array(frame["transform_matrix"])[2][3] for frame in original_transforms["frames"]]
min_z, max_z = min(z_values), max(z_values)

levels = 6
thresholds = np.linspace(min_z, max_z, levels + 1)

print(f"Min Z: {min_z}, Max Z: {max_z}")
print(f"Elevation thresholds: {thresholds}")

frames_by_level = [[] for _ in range(levels)]
for frame in original_transforms["frames"]:
    z = np.array(frame["transform_matrix"])[2][3]
    for i in range(levels):
        if thresholds[i] <= z <= thresholds[i + 1]:
            frames_by_level[i].append(frame)
            break

# Deterministic sampling: take evenly spaced frames per elevation level
for i in range(levels):
    frames = frames_by_level[i]
    total = len(frames)
    print(f"Level {i+1} has {total} frames")

    if total == 0:
        continue

    if i == 0:
        # First: take every second image
        every_second = frames[:max_images_per_level]
        selected = every_second
        
    elif total <= max_images_per_level:
        selected = frames
    else:
        indices = np.linspace(0, total - 1, max_images_per_level, dtype=int)
        selected = [frames[j] for j in indices]

    frames_by_level[i] = selected

for i, level_frames in enumerate(frames_by_level):
    level_name = f"level_{i+1}"
    level_output_path = os.path.join(base_output_root, level_name)

    if os.path.exists(level_output_path):
        print(f"{level_output_path} already exists, skipping copy.")
    else:
        print(f"Creating selective copy in {level_output_path}...")
        selective_copy(source_path, level_output_path)
        used_image_names = set(os.path.basename(frame["file_path"]) for frame in level_frames)

    # Filter images in all image folders
    for image_folder in image_folders_to_filter:
        image_folder_path = os.path.join(level_output_path, image_folder)
        if os.path.exists(image_folder_path):
            for filename in os.listdir(image_folder_path):
                if filename not in used_image_names:
                    file_to_remove = os.path.join(image_folder_path, filename)
                    os.remove(file_to_remove)
        else:
            print(f"Warning: Image folder {image_folder_path} does not exist.")

    # Overwrite transforms.json with filtered frames only
    filtered_transforms = {
        key: value if key != "frames" else level_frames
        for key, value in original_transforms.items()
    }
    transforms_output_path = os.path.join(level_output_path, "transforms.json")
    with open(transforms_output_path, "w") as f:
        json.dump(filtered_transforms, f, indent=4)

    print(f"Level {i+1}: saved {len(level_frames)} frames in {level_output_path}")

print("All elevation levels processed successfully.")

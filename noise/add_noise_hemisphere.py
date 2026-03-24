import os
import shutil
import json
import numpy as np

folder_name = "garden_manhattan_pro"       # original folder with images and transforms.json
output_name = "garden_pro"                 # output folder name base
noise_type_for_outputname = "hemisphere"  # noise type for output folder suffix

for random_point_on_the_equator in range(10):  # loop over seeds 0 to 4
    output_folder = f"{output_name}_{noise_type_for_outputname}_{random_point_on_the_equator}"

    source_path = os.path.join(os.getcwd(), folder_name)
    output_path = os.path.join("/media/efe/EFE/Datasets_AILAB/outdoor/hemisphere", output_folder)

    # Copy source to output if not exist
    if not os.path.exists(output_path):
        print(f"Copying files from {source_path} to {output_path}")
        shutil.copytree(source_path, output_path)
    else:
        print(f"Output folder {output_path} already exists, skipping copy.")

    # Load transforms.json
    transforms_path = os.path.join(output_path, "transforms.json")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Convert transform matrices to numpy arrays
    for frame in transforms["frames"]:
        frame["transform_matrix"] = np.array(frame["transform_matrix"])

    # Extract camera positions and compute mean elevation (Z)
    camera_positions = [frame["transform_matrix"][:3, 3] for frame in transforms["frames"]]
    z_values = np.array([pos[2] for pos in camera_positions])
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_mean = (z_min + z_max) / 2

    # Find frames near mean elevation (equator)
    elevation_tolerance = 0.2
    middle_frames = [
        (frame, frame["transform_matrix"][:3, 3])
        for frame in transforms["frames"]
        if abs(frame["transform_matrix"][:3, 3][2] - z_mean) < elevation_tolerance
    ]

    if not middle_frames:
        raise ValueError("No frames found near mean elevation. Try increasing elevation_tolerance.")

    # Pick one random reference frame on the equator using the current seed
    np.random.seed(random_point_on_the_equator)
    ref_frame, ref_pos = middle_frames[np.random.randint(len(middle_frames))]

    # Define scene center and reference direction vector
    scene_center = np.mean(camera_positions, axis=0)
    ref_direction = ref_pos - scene_center
    ref_direction /= np.linalg.norm(ref_direction)

    # Select all frames in front hemisphere (dot product >= 0)
    selected_frames = []
    for frame in transforms["frames"]:
        cam_pos = frame["transform_matrix"][:3, 3]
        direction = cam_pos - scene_center
        direction /= np.linalg.norm(direction)
        dot_product = np.dot(direction, ref_direction)
        if dot_product >= 0:
            selected_frames.append(frame)

    print(f"Seed {random_point_on_the_equator}: Selected {len(selected_frames)} frames in front hemisphere")

    transforms["frames"] = selected_frames

    # Convert numpy arrays back to lists for JSON serialization
    for frame in transforms["frames"]:
        if isinstance(frame["transform_matrix"], np.ndarray):
            frame["transform_matrix"] = frame["transform_matrix"].tolist()

    filtered_transforms_path = os.path.join(output_path, "transforms.json")
    with open(filtered_transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    # Remove unused images from all image folders
    used_image_names = set(os.path.basename(frame["file_path"]) for frame in transforms["frames"])
    print(f"Seed {random_point_on_the_equator}: Used image names count: {len(used_image_names)}")

    images_path_addition = ["images", "images_2", "images_4", "images_8"]
    for folder in images_path_addition:
        folder_path = os.path.join(output_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping.")
            continue

        for filename in os.listdir(folder_path):
            if filename not in used_image_names:
                file_to_delete = os.path.join(folder_path, filename)
                os.remove(file_to_delete)
                print(f"Deleted: {file_to_delete}")

    print(f"Seed {random_point_on_the_equator}: Hemisphere filtering and cleanup complete.\n")

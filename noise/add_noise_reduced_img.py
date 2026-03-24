import numpy as np
import cv2 as cv
import os
import shutil
import json

# Set seed for reproducibility
np.random.seed(42)

# Define static parameters
folder_name = "bonzai_manhattan_pro"  # original folder
source_path = os.path.join(os.getcwd(), folder_name)
output_base = "/media/efe/EFE/Datasets_AILAB/indoor/reduced_img/bonzai_manhattan_pro_reduced_img"
images_path_addition = ["images", "images_2", "images_4", "images_8"]

for every_x_image in [2, 3, 4, 5]:
    output_folder = output_base + "_" + str(every_x_image)
    output_path = os.path.join(output_folder)
    print(f"\nProcessing every {every_x_image}th image into: {output_path}")

    # Copy entire folder if it doesn't already exist
    if not os.path.exists(output_path):
        print(f"Copying {source_path} to {output_path}")
        shutil.copytree(source_path, output_path)
    else:
        print(f"Folder {output_path} already exists. Skipping copy.")

    # Collect all image names from main images folder
    im_path = os.path.join(output_path, images_path_addition[0])
    images_names = sorted(os.listdir(im_path))  # sorted to keep consistent order
    images_to_keep = images_names[::every_x_image]

    # Delete unwanted images in all folders
    for folder in images_path_addition:
        folder_path = os.path.join(output_path, folder)
        for img_file in os.listdir(folder_path):
            if img_file not in images_to_keep:
                os.remove(os.path.join(folder_path, img_file))
                print(f"Deleted {img_file} from {folder_path}")

    # Load transforms.json and filter frames
    transforms_path = os.path.join(output_path, "transforms.json")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Keep only frames with file paths that match the kept images
    filtered_frames = [frame for frame in transforms["frames"]
                       if frame["file_path"].split("/")[-1] in images_to_keep]

    # Sort frames based on image name
    filtered_frames.sort(key=lambda f: f["file_path"].split("/")[-1])

    transforms["frames"] = filtered_frames

    # Save updated transforms
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"Finished reduction for every_x_image = {every_x_image}")

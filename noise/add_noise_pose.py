import numpy as np
import cv2 as cv
import os
import shutil
import json
import torch
from jaxtyping import Float
from torch import Tensor

# List of sigmas and number of seeds
a_sigmas = [0.005, 0.01, 0.015, 0.02, 0.025]
num_seeds = 10

required_files = ["sparse_pc.ply", "transforms.json"]
image_folders = ["images_4"]  # Only images_4 now

# === Copy required files and image folders ===
def selective_copy(source_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for item in os.listdir(source_path):
        s_item_path = os.path.join(source_path, item)
        d_item_path = os.path.join(output_path, item)
        if os.path.isdir(s_item_path):
            if item in image_folders:
                shutil.copytree(s_item_path, d_item_path)
        elif item in required_files:
            shutil.copy2(s_item_path, d_item_path)

# === SE(3) exponential map ===
def exp_map_SE3(tangent_vector: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 4"]:
    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)

    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(
        near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
    )

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret

# === Main loop ===
base_output_root = "/media/efe/EFE/Datasets_AILAB/outdoor/pose"
for sigma in a_sigmas:
    for seed in range(num_seeds):
        np.random.seed(seed)

        # === Paths ===
        folder_name = "garden_manhattan_pro"
        sigma_rot = sigma
        sigma_trans = sigma
        output_name = "garden_pro_pose"
        output_folder_name = f"{output_name}_rot{sigma_rot}_trans{sigma_trans}"

        # Create seed subfolder in base output root
        seed_folder = os.path.join(base_output_root, f"seed_{seed}")
        os.makedirs(seed_folder, exist_ok=True)

        # Final output folder inside seed folder
        output_path = os.path.join(seed_folder, output_folder_name)
        source_path = os.path.join(os.getcwd(), folder_name)

        print(output_path)

        if not os.path.exists(output_path):
            print(f"Selectively copying files from {source_path} to {output_path}")
            selective_copy(source_path, output_path)
        else:
            print("Output folder already exists, skipping copy.")

        # Count images in the first image folder
        first_image_folder = os.path.join(output_path, image_folders[0])
        if os.path.exists(first_image_folder):
            num_images = len([
                f for f in os.listdir(first_image_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"Number of images in {image_folders[0]}: {num_images}")
        else:
            print(f"Warning: {first_image_folder} does not exist.")

        # === Apply pose noise ===
        transforms_path = os.path.join(output_path, "transforms.json")
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        for frame in transforms["frames"]:
            T_orig_np = np.array(frame["transform_matrix"], dtype=np.float32)

            noise = np.concatenate([
                np.random.normal(0.0, sigma_trans, size=3),
                np.random.normal(0.0, sigma_rot, size=3)
            ])
            noise_tensor = torch.tensor(noise, dtype=torch.float32).unsqueeze(0)

            T_noise = exp_map_SE3(noise_tensor)[0]

            T_noise_hom = torch.eye(4)
            T_noise_hom[:3, :4] = T_noise

            T_orig = torch.tensor(T_orig_np)
            T_noisy = T_noise_hom @ T_orig

            frame["transform_matrix"] = T_noisy.tolist()

        transforms["frames"] = sorted(transforms["frames"], key=lambda f: f["file_path"])
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=4)

        print(f"Pose noise added to transforms.json for sigma={sigma}, seed={seed}\n")

### 1.Image Subsampling
* **What it does:** Systematically reduces the density of the dataset by keeping only a fraction of the original images. By simulating a lower frame-rate capture or dropping intermediate frames, it tests a model's ability to reconstruct a scene with wider spatial gaps between camera views (few-shot reconstruction).
* **How it works:** It iterates through step sizes of 2, 3, 4, and 5. For each step size, it keeps only every $N$th image. 
* **Seed Usage:** Uses a **fixed seed (42)**. However, the randomization seed does not actually affect the output here, as the script relies purely on deterministic, sequential mathematical slicing rather than random selection.

### 2. Elevation-Based Subsampling
* **What it does:** Artificially limits the dataset to create sparse views based strictly on camera height (the Z-axis). By actively removing images and pruning the camera data, it tests a model's ability to reconstruct a scene from limited, elevation tiers (e.g., simulating drone flights at fixed altitudes).
* **How it works:** It reads the camera extrinsics from `transforms.json` to extract the Z-axis value of every frame. It calculates the absolute minimum and maximum heights and divides the scene into 6 vertical elevation "levels." For each level, it deterministically samples the frames to ensure a maximum of 33 evenly spaced images are kept per tier. Unused images are physically deleted from the output directory.
* **Seed Usage:** **Deterministic**. This script does not use a random seed. The filtering relies entirely on fixed mathematical thresholds along the Z-axis and deterministic array sampling. It will generate the exact same sparse datasets every time it is run on a given source.

### 3. Hemispherical View Filtering
* **What it does:** Restricts the dataset so all remaining cameras view the scene from roughly the same side (a single hemisphere). By physically deleting the other half of the dataset, it simulates limited capture angles.
* **How it works:** It reads `transforms.json` to calculate the center of the scene and the mean elevation (Z-axis) of all cameras, establishing an "equator." It picks one camera along this equator to act as a reference direction. Using a dot-product calculation, it keeps only the cameras that exist within the same front-facing hemisphere as that reference vector.
* **Seed Usage:** Loops through 10 seeds (0–9). The seed randomly selects the *reference camera* on the equator. Because of this, each seed produces a dataset that is cropped from a completely different directional angle (e.g., one seed might keep only the "front" cameras, while another keeps only the "left" cameras).

### 4. Gamma Noise
* **What it does:** Simulates severe lighting anomalies—specifically extreme overexposure and underexposure—on a randomized subset of the dataset. This tests a model's robustness to poorly lit environments or sudden auto-exposure failures in camera hardware.
* **How it works:** Based on the target corruption percentage (10% to 50%), the script randomly selects a subset of images from the dataset. It then splits this subset perfectly in half. The first half is subjected to a gamma correction value of `0.2` (creating blown-out, artificially bright images), while the second half receives a gamma value of `5.0` (creating heavily darkened, shadowed images).
* **Seed Usage:** Loops through 10 random seeds (0–9). For each target corruption percentage, the seed drives the random selection of *which specific images* are modified, as well as dictating which images end up in the bright group versus the dark group.

### 5. Pose Noise
* **What it does:** Simulates tracking errors, sensor drift, or poor COLMAP initializations by injecting mathematical noise directly into the camera extrinsics (poses). 
* **How it works:** It reads the original camera transformation matrices from `transforms.json`. For each camera, it samples a random 6-Degree-of-Freedom (6-DOF) noise vector (3 for translation, 3 for rotation) from a normal distribution based on predefined variance levels (sigmas ranging from `0.005` to `0.025`). Using a exponential map, it converts these raw noise vectors into valid 4x4 transformation matrices. It then multiplies the original camera poses by these noise matrices, shifting and tilting the cameras virtually.
* **Seed Usage:** Loops through 10 seeds (0–9) for every single sigma level. The seed drives the random number generator that dictates the exact translational and rotational noise applied to each matrix, creating distinct, reproducible variations of pose degradation.

### 6. Motion Blur
* **What it does:** Simulates a tracking shot effect on a randomized subset of the dataset. Foreground objects that the camera is "following" remain relatively sharp, while the distant background experiences heavy motion blur. 
* **How it works:** It computes dense optical flow between consecutive frames to determine the average direction of the camera movement. It then loads MiDaS disparity maps (`.npy` files) and normalizes them into 10 discrete layers. A directional motion blur kernel is applied, with the kernel size scaling inversely with the disparity values—meaning the deep background (low disparity) receives the largest blur, while the close foreground (high disparity) stays sharp.
* **Seed Usage:** Loops through 10 random seeds. For each seed and target corruption percentage (10% to 50%), the seed controls the random selection of *which specific images* in the sequence are subjected to the blur effect.
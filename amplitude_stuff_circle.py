import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
from scipy.stats import rankdata


def detect_circle_mask_2(img, image_path):
    # Resize to speed up (adjust scale as needed)
    scale = 0.1
    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # HSV conversion (fast color segmentation)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # Red mask (merge in one call)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 | mask2

    # Use small blur kernel (faster)
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 1)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        mask_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=20,
        minRadius=30,
        maxRadius=0
    )

    if circles is None:
        print(image_path)
        raise ValueError("No circle detected")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]

    # Scale back to original size
    x = int(x / scale)
    y = int(y / scale)
    r = int(r / scale)

    return x, y, r


def estimate_inner_radius(gray, cx, cy, r_outer, max_fraction=0.7):
    rs = np.arange(5, int(r_outer * max_fraction))
    intensities = []

    for ri in rs:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), ri, 1, -1)
        mean_val = np.mean(gray[mask == 1])
        intensities.append(mean_val)

    intensities = np.array(intensities)
    dip_index = np.argmin(intensities)
    r_inner = rs[dip_index]
    return r_inner


def detect_circle_mask(image_path, return_inner_mask=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 15)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=0
    )

    if circles is None:
        x, y, r = detect_circle_mask_2(img, image_path)
    else:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]

    # Main mask (outer circle)
    outer_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), r, 1, -1)

    # Estimate inner circle radius:
    r_inner = estimate_inner_radius(gray, x, y, r)

    # Null region mask
    inner_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(inner_mask, (x, y), r_inner, 1, -1)

    valid_mask = outer_mask - inner_mask  # 1 for valid, 0 for invalid

    if return_inner_mask:
        return x, y, r, r_inner, valid_mask
    return x, y, r, valid_mask

def create_circle_mask(shape, center_x, center_y, outer_radius, inner_radius_ratio=0):
    """
    Create a circular mask with an optional inner circle excluded.

    Parameters:
    - shape: (height, width) of the image
    - center_x, center_y: center coordinates of the circle
    - outer_radius: radius of the outer circle in pixels
    - inner_radius_ratio: float between 0 and 1, fraction of outer_radius to exclude in center

    Returns:
    - mask: boolean numpy array, True inside the ring area
    """
    # if (inner_radius_ratio == 0):
    #     return
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    inner_radius = outer_radius * inner_radius_ratio

    outer_mask = dist_from_center <= outer_radius
    inner_mask = dist_from_center <= inner_radius

    ring_mask = outer_mask & ~inner_mask
    return ring_mask

from amplitude_stuff import detect_circle_mask as dcm
im =None
def process_series_amplitude(value_to_path: dict, name: str, output_dir="amplitude_output", title="", axs=None,
                             index=0):
    os.makedirs(output_dir, exist_ok=True)

    # Detect circle from the first image
    first_path = next(iter(value_to_path.values()))
    x, y, r = dcm(first_path)

    h, w = 2 * r, 2 * r
    inner_radius = 2.5
    radius = 5
    min_map = np.full((h, w), np.inf, dtype=np.float32)
    max_map = np.full((h, w), -np.inf, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 1, -1)  # local circle mask
    sub_mask = np.zeros((h, w), dtype=np.uint8)
    sub_r = r * inner_radius / radius
    cv2.circle(sub_mask, (r, r), int(sub_r), 1, -1)
    mask = mask - sub_mask

    for img_path in value_to_path.values():
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = gray[y - r:y + r, x - r:x + r]

        valid_pixels = mask == 1
        pixel_vals = cropped.astype(np.float32)

        current_vals = np.zeros_like(cropped, dtype=np.float32)
        current_vals[valid_pixels] = pixel_vals[valid_pixels]

        min_map[valid_pixels] = np.minimum(min_map[valid_pixels], current_vals[valid_pixels])
        max_map[valid_pixels] = np.maximum(max_map[valid_pixels], current_vals[valid_pixels])

    amplitude_map = max_map - min_map
    amplitude_map[mask == 0] = np.nan  # mask outside circle
    # Visualization
    if axs is None:
        fig, ax = plt.subplots()
    else:
        ax = axs[math.floor(index / 2), index % 2]
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')

    im = ax.imshow(amplitude_map, cmap=cmap, interpolation='nearest')
    final_title = title if title else f'Amplitude Map of {name}'
    ax.set_title(final_title)
    ax.axis('off')
    if axs is None:
        plt.colorbar(label='Pixel Amplitude (Max - Min Intensity)', ax=ax)
        ax.tight_layout()

        save_path = os.path.join(output_dir, f"amplitude_{name}_map.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved amplitude map heatmap to {save_path}")

    return amplitude_map


def compute_isochromatics(value_to_path: dict, name="isochromatics", output_dir="isochromatics_output"):
    os.makedirs(output_dir, exist_ok=True)

    angles = np.array(list(value_to_path.keys()))  # in degrees
    angles_rad = np.deg2rad(angles)

    # Detect circle
    first_path = next(iter(value_to_path.values()))
    x, y, r, mask = detect_circle_mask(first_path)
    h, w = 2 * r, 2 * r
    # mask = np.zeros((h, w), dtype=np.uint8)
    # cv2.circle(mask, (r, r), r, 1, -1)
    mask = mask[y - r:y + r, x - r:x + r]  # crop the full image mask

    # Stack images
    img_stack = []
    for angle in angles:
        img = cv2.imread(value_to_path[angle])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = gray[y - r:y + r, x - r:x + r].astype(np.float32)
        img_stack.append(cropped)

    img_stack = np.stack(img_stack, axis=0)  # shape: (num_angles, H, W)

    sin2θ = np.sin(2 * angles_rad)[:, np.newaxis, np.newaxis]
    cos2θ = np.cos(2 * angles_rad)[:, np.newaxis, np.newaxis]

    S = np.sum(img_stack * sin2θ, axis=0)
    C = np.sum(img_stack * cos2θ, axis=0)

    # Amplitude of variation
    B = np.sqrt(S ** 2 + C ** 2)
    B[mask == 0] = np.nan

    # Normalize for visibility
    flat_B = B[~np.isnan(B)]
    ranks = rankdata(flat_B, method='average')  # Ranks from 1 to N
    percentiles = (ranks - 1) / (len(flat_B) - 1)  # Map to [0,1]

    # Create percentile image with same shape as B
    B_percentiles = np.full_like(B, np.nan)
    B_percentiles[~np.isnan(B)] = percentiles

    # Now apply a nonlinear mapping (e.g., arcsin to boost middle)
    B_norm = np.arcsin(2 * B_percentiles - 1) / np.pi + 0.5  # Normalized back to [0,1]

    # Visualization
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.plasma
    cmap.set_bad(color='black')
    plt.imshow(B_norm, cmap='twilight_r', interpolation='nearest')
    plt.colorbar(label='Normalized Isochromatic Intensity')
    plt.title(f'Isochromatics Map: {name}')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"2isochromatics_{name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved isochromatics map to {save_path}")

    return B_norm


fig, axs = plt.subplots(2, 2)  # 2 rows, 2 columns)
index = -1
# color_title = "Stress Axis's by Color"
radius_title = "Amplitude difference by Light color"
fig.suptitle(radius_title)
plt.tight_layout()
for prefix in ["Rin=2.5cm"]:  # "Rin=2cm",
    # for prefix in ["Rin=1cm", "Rin=1.5cm", "Rin=2cm"]:
    for name in ['green', 'red', 'blue', 'white']:
        index += 1
        images = {i: f"circles-900N/{prefix}/{i}d/{name}.jpg" for i in range(0, 91, 10)}
        print(images)
        # images = {i: f"data/{prefix}/{i}N/{name}.jpg" for i in range(100, 1801, 100)}
        # force_map = compute_isoclinics(images, f"{prefix}_{name} light", output_dir="amplitude_output")
        process_series_amplitude(images, f"{prefix}_{name} light", output_dir=prefix, title=f"{name} light", axs=axs,
                                 index=index)

        # amplitude_map = compute_isochromatics(images, f"{prefix}_{name} light", output_dir=prefix)
# plt.colorbar(label='Pixel Amplitude (Max - Min Intensity)')
if (im is not None):
    fig.colorbar(im , ax=axs.ravel().tolist(), label='Pixel Amplitude (Max - Min Intensity)')

plt.savefig("axis/amplitude_by_color_2_5.png")

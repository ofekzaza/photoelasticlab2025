import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import curve_fit
from glob import glob

# --- פונקציות עיבוד ---
from amplitude_stuff import detect_circle_mask


def load_images_by_angle(paths: dict[str, str]):
    angle_to_img = {}
    for k, v in paths.items():
        if os.path.exists(v):
            img = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
            angle_to_img[k] = img.astype(np.float32) / 255.0
    return angle_to_img


def sin_squared(phi_deg, A, theta_deg, B):
    phi_rad = np.radians(phi_deg)
    theta_rad = np.radians(theta_deg)
    return A * (np.sin(2 * (theta_rad - phi_rad))) ** 2 + B


# def create_circle_mask(shape, center_x, center_y, radius, inner_radius):
#     Y, X = np.ogrid[:shape[0], :shape[1]]
#     dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
#     mask = dist_from_center <= radius
#     h, w = 2 * r, 2 * r
#     # min_map = np.full((h, w), np.inf, dtype=np.float32)
#     # max_map = np.full((h, w), -np.inf, dtype=np.float32)
#     # mask = np.zeros((h, w), dtype=np.uint8)
#     # cv2.circle(mask, (r, r), r, 1, -1)  # local circle mask
#     sub_mask = np.zeros((X, Y), dtype=np.uint8)
#     sub_r = len(mask) * inner_radius / radius
#     cv2.circle(sub_mask, (X, Y), int(sub_r), 1, -1)
#     mask = mask - sub_mask
#     return mask

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
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    inner_radius = outer_radius * inner_radius_ratio

    outer_mask = dist_from_center <= outer_radius
    inner_mask = dist_from_center <= inner_radius

    ring_mask = outer_mask & ~inner_mask
    return ring_mask

def extract_stress_directions(images_by_angle, mask):
    angles = sorted(images_by_angle.keys())
    phi = np.radians(angles)  # (N,)
    N = len(phi)

    images = np.stack([images_by_angle[a] for a in angles], axis=-1)  # (H, W, N)
    height, width, _ = images.shape
    theta_map = np.full((height, width), np.nan, dtype=np.float32)

    coords = np.argwhere(mask)

    # Precompute 4*phi to save time
    four_phi = 4 * phi
    cos4phi = np.cos(four_phi)
    sin4phi = np.sin(four_phi)

    # Normalize weight vectors (optional)
    norm = np.sum(cos4phi ** 2)

    for i, j in coords:
        print(i, j)
        I = images[i, j, :] - np.mean(images[i, j, :])  # remove DC component
        a = np.dot(I, cos4phi) / norm
        b = np.dot(I, sin4phi) / norm

        angle_rad = 0.5 * np.arctan2(-b, -a)  # 0.5 because of 2*theta effect
        angle_deg = np.degrees(angle_rad) % 180
        theta_map[i, j] = angle_deg

    return theta_map


import torch


def fast_stress_direction_torch(images_by_angle, mask_np, device='cuda'):
    # angles: [0, 10, 20, ..., 90]
    angles = sorted(images_by_angle.keys())
    phi = torch.tensor(angles, dtype=torch.float32, device=device) * torch.pi / 180  # (N,)
    N = len(angles)

    # Stack images into tensor (H, W, N)
    image_stack = np.stack([images_by_angle[a] for a in angles], axis=-1).astype(np.float32)
    images = torch.tensor(image_stack, device=device)  # (H, W, N)
    H, W, N = images.shape

    # Flatten the image tensor to (P, N), where P = number of pixels
    images_flat = images.view(-1, N)  # (H*W, N)

    # Use mask to select only valid pixels
    mask = torch.tensor(mask_np.astype(np.bool_), device=device).view(-1)  # (H*W,)
    selected = images_flat[mask]  # (P, N), P = valid pixels only

    # Remove DC component (mean across angles)
    selected = selected - selected.mean(dim=1, keepdim=True)

    # Precompute cos(4phi) and sin(4phi)
    four_phi = 4 * phi
    cos4phi = torch.cos(four_phi)  # (N,)
    sin4phi = torch.sin(four_phi)

    # Dot products
    a = torch.sum(selected * cos4phi, dim=1)
    b = torch.sum(selected * sin4phi, dim=1)

    # Normalize (optional: norm = sum(cos4phi**2), but we skip since it's scale-invariant)
    angle_rad = 0.5 * torch.atan2(-b, -a)
    angle_deg = torch.rad2deg(angle_rad) % 180  # (P,)
    delta = 5
    valid_angles = (angle_deg <= delta) | (angle_deg >= 180 - delta)
    angle_deg_filtered = torch.where(valid_angles, angle_deg, torch.tensor(float('nan'), device=device))

    # Fill the angle map
    theta_map = torch.full((H * W,), float('nan'), dtype=torch.float32, device=device)
    theta_map[mask] = angle_deg_filtered
    # theta_map[mask] = angle_deg
    theta_map = theta_map.view(H, W)

    return theta_map.cpu().numpy()


def plot_theta_map(theta_map, save_path: str, title='Principal Stress Directions', outer_radius=0, inner_ratio=0,
                   center=None,    thickness=1/2):
    print(save_path)

    fig, ax = plt.subplots()
    im = ax.imshow(theta_map, cmap='hsv', vmin=0, vmax=180)
    if center and outer_radius > 0:
        outer_circle = patches.Circle(center, radius=outer_radius, edgecolor='black',
                                      facecolor='none', linewidth=thickness)
        ax.add_patch(outer_circle)
        # Add inner circle if needed
        if inner_ratio > 0:
            inner_circle = patches.Circle(center, radius=outer_radius * inner_ratio, edgecolor='black',
                                          facecolor='none', linewidth=thickness)
            ax.add_patch(inner_circle)

    plt.colorbar(im, ax=ax, label='Theta (degrees)')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def filter_and_dilate_theta(theta_map, min_deg=5, dilation_size=10):
    """
    Creates a binary mask of regions where theta is near 0 or 180 degrees
    and dilates the mask to make it more visible.

    Returns: dilated binary mask
    """
    # Binary mask for theta near 0° or 180°
    # mask = (theta_map <= min_deg) | (theta_map >= 180 - min_deg)
    mask = theta_map.astype(np.uint8) * 255

    # Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1))
    dilated = cv2.dilate(mask, kernel)
    highlight_rgb = np.zeros((*dilated.shape, 3), dtype=np.float32)
    highlight_rgb[dilated > 0] = [1, 1.0, 1.0]  # Red

    return highlight_rgb

def plot_theta_with_highlight(theta_map, dilated_mask, title="Filtered Theta Map", save_path=None):
    plt.figure(figsize=(10, 8))
    # plt.imshow(theta_map, cmap='hsv', vmin=0, vmax=180)

    highlight_rgb = np.zeros((*dilated_mask.shape, 3), dtype=np.float32)
    highlight_rgb[dilated_mask > 0] = [1.0, 0.0, 0.0]  # Red

    plt.imshow(highlight_rgb, alpha=0.3)

    plt.colorbar(label='Theta (degrees)')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# --- הפעלת הכל ---
# root_dir = 'circles-900N/Rin=2.5cm'  # 🔁 שנה לנתיב שלך
color = 'blue'

radius = 5

# for prefix in ["Rin=1cm", "Rin=1.5cm", "Rin=2cm"]:
colors = ['green', 'red', 'blue', 'white']
old_colors = ['green', 'red', 'blue']
for color in old_colors:#['white']:
    for i in range(2, 6):
    # for i in [4]:
        inner_radius = i / 2
        inner_radius = inner_radius if math.ceil(inner_radius) != math.floor(inner_radius) else int(inner_radius)
        prefix = f"Rin={inner_radius}cm"  # "Rin=2cm",
        # color = colors[-1]
        images_path = {i: f"circles-900N/{prefix}/{i}d/{color}.jpg" for i in range(0, 91, 10)}

        # טען תמונות
        images = load_images_by_angle(images_path)

        # מצא מעגל מתוך תמונה כלשהי (ראשונה)
        first_angle = sorted(images.keys())[0]
        # first_image_path = os.path.join(root_dir, f"{int(first_angle)}d", color, f"{color}.png")
        cx, cy, r = detect_circle_mask(images_path[0])
        radius_ration = inner_radius / radius * 0.94
        mask = create_circle_mask(images[first_angle].shape, cx, cy, r, radius_ration)

        # theta_map = extract_stress_directions(images, mask)
        theta_map = fast_stress_direction_torch(images, mask)

        # תצוגה
        dilated_mask = filter_and_dilate_theta(theta_map, min_deg=5, dilation_size=10)
        # plot_theta_with_highlight(theta_map, dilated_mask, save_path=f'axis/{prefix}/axis-{color}.png')
        plot_theta_map(dilated_mask, f'axis/{prefix}/greyscale-axis-{color}.png', title=f"{prefix} stress axis's",
                       outer_radius=r, inner_ratio=radius_ration, center=(cx, cy))
        # plot_theta_map(stress_axis, f'axis/{prefix}/axis-{color}.png', title=f"{prefix} stress axis's",
        #                outer_radius=r, inner_ratio=radius_ration, center=(cx, cy))
        # plot_theta_map(theta_map, f'axis/{prefix}/{color}.png')
        # axis_mask = detect_stress_axes(theta_map, threshold_deg=20, smooth_sigma=1.5)
        # plot_stress_axes(theta_map, axis_mask)

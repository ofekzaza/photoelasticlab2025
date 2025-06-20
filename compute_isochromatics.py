import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def detect_circle_mask(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=0
    )

    if circles is None:
        raise ValueError("No circle detected")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]
    return x, y, r

def compute_isochromatics(value_to_path: dict, name="isochromatics", output_dir="isochromatics_output"):
    os.makedirs(output_dir, exist_ok=True)

    angles = np.array(list(value_to_path.keys()))  # in degrees
    angles_rad = np.deg2rad(angles)

    # Detect circle
    first_path = next(iter(value_to_path.values()))
    x, y, r = detect_circle_mask(first_path)
    h, w = 2 * r, 2 * r
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 1, -1)

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
    B = np.sqrt(S**2 + C**2)
    B[mask == 0] = np.nan

    # Normalize for visibility
    B_norm = B / np.nanmax(B)

    # Visualization
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.plasma
    cmap.set_bad(color='black')
    plt.imshow(B_norm, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Normalized Isochromatic Intensity')
    plt.title(f'Isochromatics Map: {name}')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"isochromatics_{name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved isochromatics map to {save_path}")

    return B_norm

def plot_stress_field(phi_map, amplitude_map, mask=None, stride=10, name="stress_field", output_dir="isoclinics_output"):
    arrow_scale = stride
    """
    Quiver plot where:
    - Arrow direction = phi (from isoclinics)
    - Arrow length = amplitude (from isochromatics)
    - Arrow color = angle (phi) mapped to hue
    """
    h, w = phi_map.shape

    if mask is None:
        mask = ~np.isnan(phi_map)

    y_coords, x_coords = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')
    x_list, y_list, u_list, v_list, color_list = [], [], [], [], []

    for i, j in zip(y_coords.flatten(), x_coords.flatten()):
        if i >= h or j >= w:
            continue
        if not mask[i, j]:
            continue

        phi = phi_map[i, j]
        amp = amplitude_map[i, j]

        if amp < 0.05:
            continue

        amp *= arrow_scale
        u = np.cos(phi) * amp
        v = np.sin(phi) * amp

        # Normalize angle to [0, 1] for colormap (0 to 2π range)
        hue = (phi % (2 * np.pi)) / (2 * np.pi)

        x_list.append(j)
        y_list.append(i)
        u_list.append(u)
        v_list.append(v)
        color_list.append(hue)

    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    color_arr = np.array(color_list)

    # Colormap based on angle
    cmap = plt.cm.hsv  # or 'twilight' for smooth circularity

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray', alpha=0.05)
    q = plt.quiver(
        x_list,
        y_list,
        u_arr,
        -v_arr,
        color=cmap(color_arr),
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.0025,
        headwidth=3
    )
    plt.title("Stress Vector Field (Angle = Color, Amplitude = Length)")
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = f"{output_dir}/{name}_quiver_colored.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved color-coded stress vector field to {save_path}")
prefix = "250N"

def compute_isoclinics(value_to_path: dict, name="isoclinics", output_dir="isoclinics_output"):
    os.makedirs(output_dir, exist_ok=True)

    angles = np.array(list(value_to_path.keys()))  # in degrees
    angles_rad = np.deg2rad(angles)

    # Detect circle once
    first_path = next(iter(value_to_path.values()))
    x, y, r = detect_circle_mask(first_path)
    h, w = 2 * r, 2 * r
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 1, -1)

    # Stack all images in 3D array
    img_stack = []
    for angle in angles:
        img = cv2.imread(value_to_path[angle])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = gray[y - r:y + r, x - r:x + r].astype(np.float32)
        img_stack.append(cropped)

    img_stack = np.stack(img_stack, axis=0)  # shape: (num_angles, H, W)

    # Fit I(θ) = A + B * cos(2(θ - φ)) => least squares fit for φ
    sin2θ = np.sin(2 * angles_rad)[:, np.newaxis, np.newaxis]
    cos2θ = np.cos(2 * angles_rad)[:, np.newaxis, np.newaxis]

    S = np.sum(img_stack * sin2θ, axis=0)
    C = np.sum(img_stack * cos2θ, axis=0)

    phi_map = 0.5 * np.arctan2(S, C)  # stress angle φ (in radians)
    phi_map[mask == 0] = np.nan  # Mask outside

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(phi_map, cmap='twilight_r', interpolation='nearest',)
    plt.colorbar(label='Isoclinic Angle (radians)')
    plt.title(f'Isoclinics Map: {name}')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"isoclinics_{name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved isoclinics map to {save_path}")

    return phi_map

for name in ['green', 'red', 'blue']:
    images = {i: f"data/{prefix}/{i}d/{name}.jpg" for i in range(0, 91, 5)}
    # images = {i: f"data/{prefix}/{i}N/{name}.jpg" for i in range(100, 1801, 100)}
    # images = {i: f"week2_data/3.5cm r - 3.5 mm w/4mm D/{i}d/{name}.jpg" for i in range(0, 190   , 20)}
    #isochromatics_output
    amplitude_map = compute_isochromatics(images, f"{prefix}_{name} light", output_dir=prefix)
    phi_map = compute_isoclinics(images, f"{prefix}_{name} light", output_dir=prefix)
    plot_stress_field(phi_map, amplitude_map, stride=150, output_dir=prefix, name=f"{prefix}_{name} light")


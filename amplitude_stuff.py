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


def detect_circle_mask(image_path):
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
        return detect_circle_mask_2(img, image_path)

        # raise ValueError("No circle detected")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]
    return x, y, r


def process_series_amplitude(value_to_path: dict, radius: float, inner_radius: float, name: str,
                             output_dir="amplitude_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Detect circle from the first image
    first_path = next(iter(value_to_path.values()))
    x, y, r = detect_circle_mask(first_path)

    h, w = 2 * r, 2 * r
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
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.inferno
    cmap.set_bad(color='black')

    plt.imshow(amplitude_map, cmap='twilight_r', interpolation='nearest')
    plt.colorbar(label='Pixel Amplitude (Max - Min Intensity)')
    plt.title(f'Amplitude Map of {name}')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"amplitude_{name}_map.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved amplitude map heatmap to {save_path}")

    return amplitude_map

if __name__ == "__main__":
    radius = 5
    for i in range(2, 6):
        inner_radius = i / 2
        inner_radius = inner_radius if math.ceil(inner_radius) != math.floor(inner_radius) else int(inner_radius)
        prefix = f"Rin={inner_radius}cm"  # "Rin=2cm",
        # for prefix in ["Rin=1cm", "Rin=1.5cm", "Rin=2cm"]:
        for name in ['green', 'red', 'blue', 'white']:
            images = {i: f"circles-900N/{prefix}/{i}d/{name}.jpg" for i in range(0, 91, 10)}
            process_series_amplitude(images, name=f"{prefix}_{name} light", output_dir=prefix, radius=radius, inner_radius=inner_radius)

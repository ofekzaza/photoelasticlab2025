import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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
        raise ValueError("No circle detected")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]
    return x, y, r


def process_series(data_type: str, value_to_path: dict, name: str, output_dir="polar_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Detect circle from the first image
    first_path = next(iter(value_to_path.values()))
    x, y, r = detect_circle_mask(first_path)

    h, w = 2 * r, 2 * r
    max_map = np.full((h, w), -np.inf, dtype=np.float32)
    result_map = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 1, -1)  # local circle mask

    for value, img_path in value_to_path.items():
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cropped = gray[y - r:y + r, x - r:x + r]

        valid_pixels = mask == 1
        inverted = np.zeros_like(cropped, dtype=np.float32)

        raw_vals = cropped[valid_pixels].astype(np.float32)
        if np.max(raw_vals) > 0:
            normalized = (raw_vals / np.max(raw_vals)) * 100
            inverted_vals = 100 - normalized
            inverted[valid_pixels] = inverted_vals

            update_mask = inverted > max_map
            result_map[update_mask] = value
            max_map[update_mask] = inverted[update_mask]

    # Mask non-circle pixels
    result_map[mask == 0] = np.nan

    # Visualization
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.viridis if data_type == "force" else plt.cm.hsv
    cmap.set_bad(color='black')

    plt.imshow(result_map, cmap=cmap, interpolation='nearest')
    plt.colorbar(label=f'{data_type.capitalize()} of Max Inverted Opacity')
    plt.title(f'Max Inverted Opacity of {name} by {data_type.capitalize()}')
    plt.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{data_type}_{name}_map.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {data_type} map heatmap to {save_path}")

    return result_map


angle_to_path = {
    0: "data/3.5cm r - 3.5 mm w/4mm D/0d/DSC_0478.jpg",
    20: "data/3.5cm r - 3.5 mm w/4mm D/20d/DSC_0483.jpg",
    40: "data/3.5cm r - 3.5 mm w/4mm D/40d/DSC_0486.jpg",
    60: "data/3.5cm r - 3.5 mm w/4mm D/60d/DSC_0489.jpg",
    80: "data/3.5cm r - 3.5 mm w/4mm D/80d/DSC_0492.jpg",
    90: "data/3.5cm r - 3.5 mm w/4mm D/90d/DSC_0495.jpg"
}
force_type = "force"
angle_type = "angle"

# for name in ['green', 'red', 'blue']:
#     images = {i: f"data/3.5cm r - 3.5 mm w/4mm D/{i}d/{name}.jpg" for i in [0, 20, 40, 60, 80, 90]}
#     force_map = process_series(angle_type, images, f"{name} light", output_dir="polar_output")

prefix = "fringes"
for name in ['green', 'red', 'blue']:
    images = {i: f"data/{prefix}/{i}d/{name}.jpg" for i in range(0, 91, 5)}
    images = {i: f"data/{prefix}/{i}N/{name}.jpg" for i in range(100, 1801, 100)}
    force_map = process_series(force_type, images, f"{prefix}_{name} light", output_dir="polar_output")

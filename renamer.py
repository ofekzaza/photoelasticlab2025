import os
import cv2
import numpy as np

# Path to the main directory containing subdirectories with images
main_dir = "3.5cm r - 3.5 mm w"


def detect_dominant_color(image_path, region_size=50):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Get center region
    cx, cy = w // 2, h // 2
    half = region_size // 2
    center = img[cy - half:cy + half, cx - half:cx + half]

    # Convert to HSV
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    hue, sat, val = avg_hsv

    # OpenCV Hue is from 0–180
    # Check for BLUE first (typical blue is 100–130)
    if 70 <= hue <= 150 and sat > 50:
        return "blue"

    # Then check for RED (around 0 or 180)
    if (hue <= 30 or hue >= 140) and sat > 50:
        return "red"

    # Then check for GREEN (typical green is 35–85)
    if 35 <= hue <= 85 and sat > 50:
        return "green"

    return "unknown"


# Traverse subdirectories
for subdir, _, files in os.walk(main_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(subdir, file)
            color = detect_dominant_color(file_path)
            if color in ["red", "green", "blue"]:
                new_path = os.path.join(subdir, f"{color}.jpg")
                os.rename(file_path, new_path)
                print(f"Renamed {file_path} to {new_path}")

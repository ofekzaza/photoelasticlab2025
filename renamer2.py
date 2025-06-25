import os
import numpy as np
from PIL import Image


# def get_dominant_color(image_path):
#     """Classify image color as red, green, blue, or white using RGB intensity and similarity."""
#     image = Image.open(image_path).convert("RGB")
#     np_image = np.array(image)
#     avg_color = np.mean(np.mean(np_image, axis=0), axis=0)
#     r, g, b = avg_color
#
#     # Check if it's white (all channels are high and close to each other)
#     if r > 180 and g > 180 and b > 180 and max(r, g, b) - min(r, g, b) < 30:
#         return "white"
#
#     # Otherwise, return dominant color
#     if r > g and r > b:
#         return "red"
#     elif g > r and g > b:
#         return "green"
#     else:
#         return "blue"

def get_dominant_color(image_path):
    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)
    avg_color = np.mean(np.mean(np_image, axis=0), axis=0)
    r, g, b = avg_color
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    brightness = np.mean([r, g, b])

    # Consider "white/gray" if the RGB values are close and brightness is mid-to-high
    if diff < 15 and brightness > 50:
        return "white"

    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    else:
        return "blue"


def classify_and_rename_images_in_subdirs(base_path):
    """Process all subdirectories, renaming 4 images per folder according to dominant color."""
    renamed_files = {}

    for root, dirs, files in os.walk(base_path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) == 4:
            color_count = {"red": 0, "green": 0, "blue": 0, "white": 0}

            for img_file in image_files:
                full_path = os.path.join(root, img_file)
                color = get_dominant_color(full_path)

                # Prevent name clashes
                suffix = f"_{color_count[color]}" if color_count[color] > 0 else ""
                new_name = f"{color}{suffix}.jpg"
                new_path = os.path.join(root, new_name)

                os.rename(full_path, new_path)
                renamed_files[full_path] = new_path
                color_count[color] += 1

    return renamed_files

# Set your base folder here
base_directory = "2.5cm r - 1cm w"

# Run the renaming process
results = classify_and_rename_images_in_subdirs(base_directory)

# Print summary of changes
for old_path, new_path in results.items():
    print(f"Renamed: {old_path} -> {new_path}")

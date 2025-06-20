import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(image_path, output_dir=None, show_plot=False):
    # Load and prepare image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 15)

    # Detect circle
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
        print(f"No circle detected in {image_path}")
        return

    # Use first detected circle
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]

    # Create full image mask and crop circle region
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    circle_pixels = gray[y - r:y + r, x - r:x + r]
    mask_crop = mask[y - r:y + r, x - r:x + r]

    # Extract valid pixels
    valid_pixels = mask_crop > 0
    opacities = np.zeros_like(circle_pixels, dtype=np.float32)

    raw_values = circle_pixels[valid_pixels].astype(np.float32)

    if raw_values.size > 0 and np.max(raw_values) > 0:
        normalized = (raw_values / np.max(raw_values)) * 100
        inverted = 100 - normalized
        opacities[valid_pixels] = inverted

    # Plot heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(opacities, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Inverted Opacity (0-100)')
    plt.title("Inverted Opacity Heatmap (Inside Circle Only)")
    plt.axis('off')
    plt.tight_layout()

    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + '_clean_inverted_heatmap.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


process_image('data/3.5cm r - 3.5 mm w/4mm F/600N/DSC_0557.jpg', output_dir="outputs", show_plot=False)
process_image("data/2.5cm r - 1cm w/450N/DSC_0628.jpg", output_dir="outputs", show_plot=False)
# For batch processing:
# import glob
# for img_path in glob.glob("images/*.jpg"):
#     process_image(img_path, output_dir="heatmaps")

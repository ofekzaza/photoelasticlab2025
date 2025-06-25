import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 5.0
P = 900.0
n = 400
tolerance_deg = 5  # increased degrees for isoclinic fringe thickness
rotation_angle_deg = 45  # rotation angle for visualizing the result only


def simulate():
    # Derived values
    tolerance = np.radians(tolerance_deg)
    rotation_angle_rad = np.radians(rotation_angle_deg)
    # Create grid
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    X, Y = np.meshgrid(x, y)
    # Normalized coordinates
    x_r = X / R
    y_r = Y / R
    r = np.sqrt(X ** 2 + Y ** 2)
    mask = r > R
    # Apply Brazilian disk stress field (horizontal loading)
    sigma_xx = (P / np.pi) * ((1 - (x_r ** 2 - y_r ** 2) / (x_r ** 2 + y_r ** 2)) / (x_r ** 2 + y_r ** 2))
    sigma_yy = -(2 * P / np.pi) * (x_r * y_r) / (x_r ** 2 + y_r ** 2) ** 2
    sigma_xy = -(P / np.pi) * ((x_r ** 2 - y_r ** 2) / (x_r ** 2 + y_r ** 2) ** 2)
    # Avoid singularity
    sigma_xx[r == 0] = 0
    sigma_yy[r == 0] = 0
    sigma_xy[r == 0] = 0
    # Principal stress angle in radians
    theta_p = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
    theta_p[mask] = np.nan
    # Crossed polarizers: find isoclinics (principal axis aligns with polarizer)
    theta_mod = np.abs(theta_p) % (np.pi / 2)
    isoclinic_mask = (theta_mod < tolerance) | (np.abs(theta_mod - (np.pi / 2)) < tolerance)
    isoclinic_mask[mask] = False
    # Rotate result image by -45° for visualization only
    X_rot = X * np.cos(rotation_angle_rad) - Y * np.sin(rotation_angle_rad)
    Y_rot = X * np.sin(rotation_angle_rad) + Y * np.cos(rotation_angle_rad)
    return X_rot, Y_rot, isoclinic_mask


X_rot, Y_rot, isoclinic_mask = simulate()

# Plot rotated isoclinics (fix axis directions)
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.pcolormesh(X_rot, Y_rot, isoclinic_mask, cmap='gray_r', shading='auto')
# ax.set_title("Isoclinics Rotated by -45° (Visualization Only)")
# ax.set_xlabel("x (horizontal)")
# ax.set_ylabel("y (vertical)")
# # ax.set_aspect('equal')
# # plt.grid(True)
# # plt.tight_layout()
# plt.show()

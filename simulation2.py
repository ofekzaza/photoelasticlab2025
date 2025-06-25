import numpy as np
import matplotlib.pyplot as plt

# Geometry
outer_radius = 5.0  # cm
inner_radius = 2.5  # cm
N = 400  # Grid resolution

# Coordinate grid
x = np.linspace(-outer_radius, outer_radius, N)
y = np.linspace(-outer_radius, outer_radius, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
mask = (R >= inner_radius) & (R <= outer_radius)

# Dipole-like stress field (approximate from opposing point forces)
F = 1.0  # Arbitrary unit force
epsilon = 0.1  # Regularization
sigma_xx = np.zeros_like(R)
sigma_yy = np.zeros_like(R)
tau_xy = np.zeros_like(R)
sigma_xx[mask] = -F * (X[mask]**2 - Y[mask]**2) / (R[mask]**4 + epsilon)
sigma_yy[mask] = F * (X[mask]**2 - Y[mask]**2) / (R[mask]**4 + epsilon)
tau_xy[mask] = -2 * F * X[mask] * Y[mask] / (R[mask]**4 + epsilon)

# Principal stress angle (in degrees)
phi = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)
phi_deg = np.degrees(phi)
phi_deg[~mask] = np.nan

# Get average angle along x-axis (center row)
center_y_index = N // 2
phi_x_axis = phi_deg[center_y_index, :]
phi_ref = np.nanmean(phi_x_axis)

# Highlight regions where the angle is ≈ same as along x-axis
tolerance = 10.0  # degrees
phi_deg = np.abs(phi_deg)
degree_mask = phi_deg > 90
normalized_phi_deg = phi_deg
normalized_phi_deg[degree_mask] -= 0
plt.imshow(normalized_phi_deg)
plt.colorbar()
plt.show()
angle_match = (np.abs(normalized_phi_deg) <= tolerance)

# Plot matching regions
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(X, Y, angle_match, levels=[0.5, 1.5], colors=['#1f77b4'], alpha=0.7)

# Inner and outer circle
ax.add_artist(plt.Circle((0, 0), inner_radius, color='black', fill=False, linewidth=2))
ax.add_artist(plt.Circle((0, 0), outer_radius, color='black', fill=False, linewidth=2))

# Point load locations
ax.plot([0, outer_radius], [0, 0], 'ro', label='Load Points')

ax.set_aspect('equal')
ax.set_title(f"Regions Where φ ≈ {phi_ref:.1f}° (Same as y = 0 Axis)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.legend()
plt.tight_layout()
plt.show()

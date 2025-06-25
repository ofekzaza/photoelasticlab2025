import numpy as np
import matplotlib.pyplot as plt

# Geometry
outer_radius = 5.0  # cm
inner_radius = 1.5  # cm
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

phi_degrees = phi_deg

# Plot isoclinics with degrees
fig, ax = plt.subplots(figsize=(8, 8))
num_isoclinics = 20
levels = np.linspace(-90, 90, num_isoclinics)
isoclinic_lines = ax.contour(X, Y, phi_degrees, levels=levels, cmap='twilight')

# Add inner and outer disc boundaries
ax.add_artist(plt.Circle((0, 0), inner_radius, color='black', fill=False, linewidth=2))
ax.add_artist(plt.Circle((0, 0), outer_radius, color='black', fill=False, linewidth=2))

# Add point load markers at x = 0 and x = R, y = 0
ax.plot([0, outer_radius], [0, 0], 'ro', label='Load Points')

# Formatting
ax.set_aspect('equal')
ax.set_title("Isoclinics: Principal Stress Direction φ (Degrees)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
plt.colorbar(isoclinic_lines, ax=ax, label='Angle φ (degrees)')
plt.tight_layout()
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- user input ----
print("Try: x**2 + y**2  OR  np.sin(x) * np.cos(y)")
input_str = input("Enter a scalar function f(x,y): ") or "x**2 + y**2"

# ---- convert string to function ----
def f(x, y):
    safe_dict = {"x": x, "y": y, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}
    return eval(input_str, {"__builtins__": None}, safe_dict)

# ---- grid ----
res = 40 
x_vec = np.linspace(-5, 5, res)
y_vec = np.linspace(-5, 5, res)
X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

# ---- scalar field ----
Z = f(X, Y)

# ---- gradient ----
dZdx, dZdy = np.gradient(Z, x_vec, y_vec, edge_order=2)

# ---- Laplacian ----
d2Zdx2, _ = np.gradient(dZdx, x_vec, y_vec, edge_order=2)
_, d2Zdy2 = np.gradient(dZdy, x_vec, y_vec, edge_order=2)

# Rounding kills the 1e-14 floating point "glitches"
laplacian = np.round(d2Zdx2 + d2Zdy2, 6)

# ---- arrow density control ----
skip = (slice(None, None, 3), slice(None, None, 3))

# ---- plot setup (2x2 Grid) ----
fig = plt.figure(figsize=(13, 11))

# Helper to keep 2D plots SQUARE and font small
def format_2d(ax, data, title, cmap):
    d_min, d_max = np.min(data), np.max(data)
    
    # Check if data is constant to prevent "noise contouring"
    if np.isclose(d_min, d_max, atol=1e-4):
        vmin, vmax = d_min - 0.5, d_max + 0.5
        im = ax.contourf(X, Y, data, levels=1, cmap=cmap, vmin=vmin, vmax=vmax)
        ticks = [d_min] 
    else:
        im = ax.contourf(X, Y, data, levels=40, cmap=cmap)
        ticks = None

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, pad=10)
    ax.tick_params(labelsize=8)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=7)
    return im

# 1) Gradient (2D)
ax1 = fig.add_subplot(2, 2, 1)
format_2d(ax1, Z, f"Gradient: {input_str}", 'viridis')
ax1.quiver(X[skip], Y[skip], dZdx[skip], dZdy[skip], scale=20)

# 2) Scalar (3D)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax2.contour(X, Y, Z, levels=20, offset=np.min(Z), cmap='viridis', alpha=0.5)
ax2.set_title("3D Scalar Field", fontsize=10)
ax2.tick_params(labelsize=7)

# 3) Laplacian (2D)
ax3 = fig.add_subplot(2, 2, 3)
format_2d(ax3, laplacian, r"Laplacian $\Delta f$", 'coolwarm')

# 4) Laplacian (3D)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(X, Y, laplacian, cmap='coolwarm', edgecolor='none', alpha=0.8)

# Floor contours only if there is variation
if not np.isclose(np.min(laplacian), np.max(laplacian)):
    ax4.contour(X, Y, laplacian, levels=15, offset=np.min(laplacian), cmap='coolwarm', alpha=0.5)

# Fix Z-limit for constant fields to prevent vertical glitching
if np.isclose(np.min(laplacian), np.max(laplacian)):
    avg_val = np.mean(laplacian)
    ax4.set_zlim(avg_val - 1, avg_val + 1)

ax4.set_title("3D Laplacian View", fontsize=10)
ax4.tick_params(labelsize=7)

plt.tight_layout(pad=3.0)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- user input ----
print("Example: Fx = -y, Fy = x")
Fx_str = input("Enter Fx(x,y): ") or "-y"
Fy_str = input("Enter Fy(x,y): ") or "x"

def eval_func(func_str, x_grid, y_grid):
    safe_dict = {"x": x_grid, "y": y_grid, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}
    return eval(func_str, {"__builtins__": None}, safe_dict)

# ---- grid setup ----
res = 60 
axis = 5 
x_coords = np.linspace(-axis, axis, res)       
y_coords = np.linspace(-axis, axis, res)
X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]

# ---- math ----
U, V = eval_func(Fx_str, X, Y), eval_func(Fy_str, X, Y)
F_mag = np.sqrt(U**2 + V**2) 

# Derivatives with high-order edges
dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
dVdy, dVdx = np.gradient(V, dy, dx, edge_order=2)

# Laplacian components
_, d2U_dx2 = np.gradient(dUdx, dy, dx, edge_order=2); d2U_dy2, _ = np.gradient(dUdy, dy, dx, edge_order=2)
_, d2V_dx2 = np.gradient(dVdx, dy, dx, edge_order=2); d2V_dy2, _ = np.gradient(dVdy, dy, dx, edge_order=2)

# Precise rounding to kill glitches
div = np.round(dUdx + dVdy, 6)
curl = np.round(dVdx - dUdy, 6)
Lx, Ly = np.round(d2U_dx2 + d2U_dy2, 6), np.round(d2V_dx2 + d2V_dy2, 6)
L_mag = np.round(np.sqrt(Lx**2 + Ly**2), 6)

# ---- plotting helpers ----
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
skip = (slice(None, None, 3), slice(None, None, 3))

def format_2d(ax, data, title, cmap):
    d_min, d_max = np.min(data), np.max(data)
    
    # Check if data is constant (The Scalar script logic)
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

# ---- 1. Field Magnitude ----
format_2d(axs[0, 0], F_mag, fr"Field $\mathbf{{F}}$: [{Fx_str}, {Fy_str}]", "viridis")
axs[0, 0].quiver(X[skip], Y[skip], U[skip], V[skip], color='white', alpha=0.7)

# ---- 2. Divergence ----
format_2d(axs[0, 1], div, r"Divergence $\nabla \cdot \mathbf{F}$", "RdBu_r")

# ---- 3. Curl ----
format_2d(axs[1, 0], curl, r"Curl $\nabla \times \mathbf{F}$", "PuOr")

# ---- 4. Laplacian ----
format_2d(axs[1, 1], L_mag, r"Laplacian $\nabla^2 \mathbf{F}$", "magma")
if np.max(L_mag) > 1e-5:
    axs[1, 1].quiver(X[skip], Y[skip], Lx[skip], Ly[skip], color='white', alpha=0.6)

# Final cleanup
for ax in axs.flat:
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)

plt.tight_layout(pad=3.0)
plt.show()

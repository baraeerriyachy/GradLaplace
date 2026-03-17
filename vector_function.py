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

# Floating point noise cleaning
div = np.round(dUdx + dVdy, 10)
curl = np.round(dVdx - dUdy, 10)
Lx, Ly = np.round(d2U_dx2 + d2U_dy2, 10), np.round(d2V_dx2 + d2V_dy2, 10)
L_mag = np.round(np.sqrt(Lx**2 + Ly**2), 10)

# ---- plotting ----
fig, axs = plt.subplots(2, 2, figsize=(11, 9))

def plot_with_colorbar(ax, data, title, cmap, symmetric=True):
    d_min, d_max = np.min(data), np.max(data)
    
    if np.isclose(d_min, d_max, atol=1e-8):
        levels = np.linspace(d_min - 1, d_max + 1, 51)
        vmin, vmax = d_min - 1, d_max + 1
    else:
        levels = 50
        if symmetric:
            limit = max(abs(d_min), abs(d_max))
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = d_min, d_max
            
    im = ax.contourf(X, Y, data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Smaller title
    ax.set_title(title, fontsize=9, pad=6)
    
    # Thinner colorbar to maximize subplot width
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)

skip = (slice(None, None, 4), slice(None, None, 4))

# 1. Field Magnitude + Arrows (Default Arrow Magnitude)
plot_with_colorbar(axs[0, 0], F_mag, fr"Field $\mathbf{{F}}$=[{Fx_str}$,{Fy_str}$]", "viridis", symmetric=False)
axs[0, 0].quiver(X[skip], Y[skip], U[skip], V[skip], color='white', alpha=0.7)

# 2. Divergence
plot_with_colorbar(axs[0, 1], div, r"Divergence $\nabla \cdot \mathbf{F}$", "RdBu_r")

# 3. Curl
plot_with_colorbar(axs[1, 0], curl, r"Curl $\nabla \times \mathbf{F}$", "PuOr")

# 4. Vector Laplacian Magnitude + Arrows (Default Arrow Magnitude)
plot_with_colorbar(axs[1, 1], L_mag, r"Laplacian $\nabla^2 \mathbf{F}$", "magma", symmetric=False)
if np.max(L_mag) > 1e-7:
    axs[1, 1].quiver(X[skip], Y[skip], Lx[skip], Ly[skip], color='white', alpha=0.6)

# Font and Subplot Spacing
for ax in axs.flat:
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)

plt.tight_layout(pad=1.5)
plt.show()

import numpy as np
import os
import matplotlib.pyplot as plt

# Grid for theta parameters
theta1 = np.linspace(-3, 3, 400)
theta2 = np.linspace(-3, 3, 400)
T1, T2 = np.meshgrid(theta1, theta2)

# Unconstrained minimizer (center of J)
opt1, opt2 = 2.0, 1.0

# Cost function J contours centered at (opt1, opt2)
J = ((T1 - opt1) / 2)**2 + (T2 - opt2)**2

# Elastic Net parameters
alpha = 0.5  # weight between L1 and L2
penalty_en = alpha * (np.abs(T1) + np.abs(T2)) + (1 - alpha) * (T1**2 + T2**2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ridge (L2)
cs0 = axes[0].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8))
axes[0].clabel(cs0)
theta = np.linspace(0, 2 * np.pi, 200)
axes[0].plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2)  # L2 ball at origin
axes[0].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[0].set_title("Ridge (L2)")
axes[0].set_xlabel("θ₁")
axes[0].set_ylabel("θ₂")
axes[0].legend()
axes[0].axhline(0, linewidth=0.5)
axes[0].axvline(0, linewidth=0.5)

# Lasso (L1)
cs1 = axes[1].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8))
axes[1].clabel(cs1)
x = np.linspace(-1, 1, 200)
axes[1].plot(x, 1 - np.abs(x), 'r-', linewidth=2)
axes[1].plot(x, -1 + np.abs(x), 'r-', linewidth=2)
axes[1].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[1].set_title("Lasso (L1)")
axes[1].set_xlabel("θ₁")
axes[1].set_ylabel("θ₂")
axes[1].legend()
axes[1].axhline(0, linewidth=0.5)
axes[1].axvline(0, linewidth=0.5)

# Elastic Net
cs2 = axes[2].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8))
axes[2].clabel(cs2)
ce = axes[2].contour(T1, T2, penalty_en, levels=[1], colors='r', linewidths=2)
axes[2].clabel(ce, fmt={1: 'α·∥θ∥₁+(1-α)·∥θ∥₂²=1'})
axes[2].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[2].set_title("Elastic Net (α=0.5)")
axes[2].set_xlabel("θ₁")
axes[2].set_ylabel("θ₂")
axes[2].legend()
axes[2].axhline(0, linewidth=0.5)
axes[2].axvline(0, linewidth=0.5)

plt.tight_layout()

# ensure the directory exists
os.makedirs("figs", exist_ok=True)

# save the current figure (or use fig.savefig if you kept a reference to it)
plt.savefig("figs/contours.png", dpi=300, bbox_inches="tight")

plt.show()

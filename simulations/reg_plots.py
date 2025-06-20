import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (18, 5),
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

theta1 = np.linspace(-2, 5, 400)
theta2 = np.linspace(-2, 5, 400)
T1, T2 = np.meshgrid(theta1, theta2)

opt1, opt2 = 2.5, 1.5
J = ((T1 - opt1) / 2)**2 + (T2 - opt2)**2

gamma = 0.5
penalty_en = gamma * (np.abs(T1) + np.abs(T2)) + (1 - gamma) * (T1**2 + T2**2)

# plot
fig, axes = plt.subplots(1, 3)

# Lasso (l1)
axes[0].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8), cmap='gray')
x = np.linspace(-1, 1, 200)
axes[0].plot(x, 1 - np.abs(x), 'r-', linewidth=2)
axes[0].plot(x, -1 + np.abs(x), 'r-', linewidth=2)
axes[0].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[0].set_title(r"Lasso ($\ell_1$)")
axes[0].set_xlabel(r"$\theta_1$")
axes[0].set_ylabel(r"$\theta_2$")
axes[0].legend()
axes[0].axhline(0, linewidth=0.5)
axes[0].axvline(0, linewidth=0.5)

# Ridge (l2)
axes[1].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8), cmap='gray')
theta = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2)
axes[1].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[1].set_title(r"Ridge ($\ell_2$)")
axes[1].set_xlabel(r"$\theta_1$")
axes[1].set_ylabel(r"$\theta_2$")
axes[1].legend()
axes[1].axhline(0, linewidth=0.5)
axes[1].axvline(0, linewidth=0.5)

# Elastic Net
axes[2].contour(T1, T2, J, levels=np.linspace(0.1, 4, 8), cmap='gray')
ce = axes[2].contour(T1, T2, penalty_en, levels=[1], colors='r', linewidths=2)
axes[2].clabel(ce, fmt={1: r'\lambda_{\ell_1}\vert \vert \theta \vert \vert_1 + \lambda_{\ell_2}\vert \vert \theta \vert \vert_2^2 = 1'}, fontsize=12)
axes[2].scatter(opt1, opt2, color='blue', marker='x', s=100, label='Optimum')
axes[2].set_title(r"Elastic Net ($\lambda_{\ell_1}=\lambda_{\ell_2}$)")
axes[2].set_xlabel(r"$\theta_1$")
axes[2].set_ylabel(r"$\theta_2$")
axes[2].legend()
axes[2].axhline(0, linewidth=0.5)
axes[2].axvline(0, linewidth=0.5)

plt.savefig(r"C:\Users\AskeElsøeEngmark\PycharmProjects\MastersThesis\figs\contours.png", dpi=300, bbox_inches='tight')

plt.show()

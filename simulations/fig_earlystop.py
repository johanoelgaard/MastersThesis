import numpy as np
import matplotlib.pyplot as plt

# Global settings
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# 1) Contour setup
theta1 = np.linspace(-2, 6, 400)
theta2 = np.linspace(-2, 6, 400)
T1, T2 = np.meshgrid(theta1, theta2)
opt = np.array([4.0, 3.0])
J = ((T1 - opt[0]) / 2)**2 + (T2 - opt[1])**2

def grad(theta):
    return np.array([(theta[0] - opt[0]) / 2,
                     2 * (theta[1] - opt[1])])

# 2) Gradient descent + early stopping simulation
eta = 0.05      # smaller step size for wider spread
max_iters = 200
patience = 3

theta = np.array([0.0, 0.0])
path = [theta.copy()]
train_loss = []
val_loss = []
best_val = np.inf
wait = 0
best_idx = 0
stop_idx = max_iters
np.random.seed(42)

for i in range(1, max_iters + 1):
    # gradient update
    theta = theta - eta * grad(theta)
    path.append(theta.copy())

    # training loss
    t_loss = ((theta - opt) * np.array([0.5, 1.0]))**2
    t_loss = t_loss.sum()
    train_loss.append(t_loss)

    # validation loss: train + 0.2 noise + slight bump past iter 8
    noise = np.random.normal(scale=0.1)
    v_loss = t_loss + 0.2 + noise
    if i > 8:
        v_loss += 0.04 * (i - 8)
    val_loss.append(v_loss)

    # early stopping on validation
    if v_loss < best_val:
        best_val = v_loss
        best_idx = i
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            stop_idx = i
            break

# trim to stopping
total_iters = stop_idx
path = np.array(path[:total_iters+1])
iters = np.arange(1, total_iters+1)
train_loss = train_loss[:total_iters]
val_loss = val_loss[:total_iters]

# 3) Plotting
fig, (ax1, ax2) = plt.subplots(1, 2)

# Left: full contour + GD path
ax1.contour(T1, T2, J, levels=np.linspace(0.1, 6, 10), cmap='gray')
ax1.plot(path[:,0], path[:,1], '-', lw=1.5, label='Gradient Descent path')
# points in gray
ax1.scatter(path[1:,0], path[1:,1], s=30, color='gray')
# highlight best iteration
best_point = path[best_idx]
ax1.scatter(best_point[0], best_point[1], s=100, color='red',
            label=f'Best Iter {best_idx}')
# optimum marker
ax1.scatter(opt[0], opt[1], color='blue', marker='x', s=100,
            label='Optimum')
ax1.set_title('Gradient Descent Path on Loss Contours')
ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.axhline(0, lw=0.5)
ax1.axvline(0, lw=0.5)
ax1.legend()

# Right: loss curves + markers
ax2.plot(iters, train_loss, label='Training Loss')
ax2.plot(iters, val_loss, label='Validation Loss')
ax2.axvline(best_idx, color='green', linestyle='--',
            label=f'Best Model at iter {best_idx}')
ax2.axvline(stop_idx, color='red', linestyle=':',
            label=f'Patience reached at iter {stop_idx}')
ax2.set_title('Training & Validation Loss')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\AskeElsøeEngmark\PycharmProjects\MastersThesis\figs\earlystop.png", dpi=300, bbox_inches='tight')
plt.show()

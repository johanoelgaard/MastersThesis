import numpy as np

# reproducible randomness
rng = np.random.default_rng(seed=42)

# 1) draw inputs: 5 rows, values in [-3, 3]
X = rng.integers(-3, 4, size=(8, 3))   # shape (5,3)
x1, x2, x3 = X.T                       # unpack columns

# 2) Gaussian noise, sigma = 0.1
noise = rng.normal(0, 0.1, size=8)

# 3) compute targets
y = 2*x1 - 3*x2 + 0.5*x3 + noise

# 4) pretty-print
print(" No.   x1   x2   x3      y")
print("----  ---  ---  ---  --------")
for i, (a, b, c, t) in enumerate(zip(x1, x2, x3, y), start=1):
    print(f"{i:>3}  {a:>3}  {b:>3}  {c:>3}  {t:8.4f}")


#### Kaiming init draw ####
alpha  = 0.1                        # Leaky‐ReLU negative slope
fan_in = 3                          # each hidden neuron sees 3 inputs

# ─── Hidden‐layer (Leaky ReLU) init ───────────────────────────────
# variance = 2 / ((1+alpha^2) * fan_in)
sigma_hidden = np.sqrt(2.0 / ((1 + alpha**2) * fan_in))

w1 = rng.standard_normal(3) * sigma_hidden  # weights for hidden neuron 1
w2 = rng.standard_normal(3) * sigma_hidden  # weights for hidden neuron 2
w3 = rng.standard_normal(3) * sigma_hidden  # weights for hidden neuron 3

# ─── Output‐layer (identity) init ─────────────────────────────────
# Glorot normal: variance = 1 / fan_in
sigma_beta = np.sqrt(1.0 / fan_in)

beta = rng.standard_normal(3) * sigma_beta  # output weight vector

# ─── Pretty‐print ─────────────────────────────────────────────────
print("Hidden‐layer weight vectors (w1, w2, w3):")
for m, w in enumerate((w1, w2, w3), start=1):
    print(f"  w{m} =", w)

print("\nOutput weight vector β:")
print("  beta =", beta)
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

fan_in = 3
sigma  = (2.0 / fan_in) ** 0.5           # √(2/3)

# ----- hidden-layer weights: three row-vectors ----------------------
w1 = rng.standard_normal(3) * sigma      #  w_11, w_12, w_13
w2 = rng.standard_normal(3) * sigma      #  w_21, w_22, w_23
w3 = rng.standard_normal(3) * sigma      #  w_31, w_32, w_33

# ----- output-layer weights ----------------------------------------
beta = rng.standard_normal(3) * sigma    # β1, β2, β3

# optional: biases (kept zero)
b_hidden = np.zeros(3)
b_out    = 0.0

# pretty-print
print("Hidden-layer weight vectors (row notation):")
for m, w in enumerate((w1, w2, w3), start=1):
    print(f"w{m} =", w)

print("\nOutput weight vector:")
print("beta =", beta)
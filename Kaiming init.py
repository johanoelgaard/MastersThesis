import numpy as np

rng = np.random.default_rng(seed=42)      # sæt seed for reproducerbarhed

fan_in1 = 3                # første lag: 3 indgående forbindelser
sigma1 = np.sqrt(2.0 / fan_in1)

W1 = rng.standard_normal((3, 3)) * sigma1   # (out, in)
b1 = np.zeros((3, 1))

fan_in2 = 3                # andet lag modtager 3 aktiveringer
sigma2 = np.sqrt(2.0 / fan_in2)

W2 = rng.standard_normal((1, 3)) * sigma2
b2 = np.zeros((1, 1))

print(W1) # til gemt lag
print(W2) # til outputlag
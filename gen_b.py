import numpy as np

b0 = np.round(np.random.uniform(-1, 2, size=(1, 20)), 1)
b1 = np.round(np.random.uniform(-1, 2, size=(1, 20)), 1)
print(b0.tolist(), b1.tolist())
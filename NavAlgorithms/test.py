import numpy as np
N=2
num_bots = 3
U0 = np.random.rand(num_bots, N, 3)
U0[:,:,2] = np.zeros((num_bots, N))
print(U0)

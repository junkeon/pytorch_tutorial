import numpy as np
import torch

def generate_sine(file_name='traindata.pt', seed=2):
    np.random.seed(seed)

    T = 20
    L = 1000
    N = 500

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float32')
    torch.save(data, open(file_name, 'wb'))
# import os
# import sys
# sys.path.insert(0, os.path.join(os.getcwd(), "../"))
# sys.path.insert(0, os.getcwd())

import torch
from modeling_rnnt import StackTime

if __name__ == '__main__':
    N = 1
    C = 2
    T = 2
    factor = 2
    x = torch.linspace(0, N*C*T-1, N*C*T, dtype=torch.int8)
    x = x.reshape((T, N, C))
    print(f'T:{T}, N:{N}, C:{C}')
    print('--x:', x, x.shape)
    x_lens = torch.ones(N)
    st = StackTime(factor)
    y, y_lens = st(x, x_lens)
    print('--y', y, y.shape)

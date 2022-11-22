import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "models"))
sys.path.insert(0, os.getcwd())

import torch
from models.modeling_rnnt import StackTime
torch.set_printoptions(precision=4,sci_mode=False)

if __name__ == '__main__':
    T = 1
    N = 2
    C = 69
    factor = 2
    x_dtype = torch.int8
    print(f'T:{T}, N:{N}, C:{C}, F:{factor}')

    # x = torch.linspace(0, N*C*T-1, N*C*T, dtype=x_dtype).reshape((T, N, C))
    x = torch.randint(0, 127, (T, N, C), dtype=torch.int8)

    x_lens = torch.full((N, 1), T, dtype=torch.int32)

    print(f'--x:{x.dtype}:{x.shape}\n{x}')
    print(f'--x_lens:{x_lens.dtype}:{x_lens.shape}\n{x_lens}')
    st = StackTime(factor, "f32")
    y, y_lens = st(x, x_lens)
    print(f'--y:{y.dtype}:{y.shape}\n{y}')
    print(f'--y_lens:{y_lens.dtype}:{y_lens.shape}\n{y_lens}')

    print('-'*30)

    # print(f'--x:{x.dtype}:{x.shape}\n{x}')
    # print(f'--x_lens:{x_lens.dtype}:{x_lens.shape}\n{x_lens}')
    y1, y_lens1 = torch.ops.intel_mlperf.stack_time(x, x_lens, factor)
    print(f'--y1:{y1.dtype}:{y1.shape}\n{y1}')
    print(f'--y_lens1:{y_lens1.dtype}:{y_lens1.shape}\n{y_lens1}')
    print(y == y1)
    print(y_lens == y_lens1)

    # N = 1
    # C = 2
    # T = 12
    # x0 = torch.linspace(0, N*C*T-1, N*C*T)
    # x1 = x0.reshape((N, C, T))
    # y1 = torch.ops.intel_mlperf.frame_splicing(x, 3)
    # print('--x1:', x1)
    # print('--y1:', y1)
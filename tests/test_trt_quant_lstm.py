import tensorrt as trt
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn


if __name__ == "_main__":
    L = 2
    T = 5
    N = 3
    IC = 10
    OC = 20

    x = torch.randn(T, N, IC)
    hx = torch.randn(L, N, OC)
    cx = torch.randn(L, N, OC)

    lstm = quant_nn.QuantLSTM(IC, OC, L)
    y, (hy, cy) = lstm(x, (hx, cx))

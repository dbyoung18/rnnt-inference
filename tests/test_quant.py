import torch
from torch.nn import Linear, LSTM
from quant_modules import QuantLinear, QuantLSTM

def test_quant_linear(in_features, out_features, batch_size):
    linear = QuantLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    y = linear(x)
    return y

def test_quant_lstm(input_size, hidden_size, num_layers, batch_size, seq_length):
    with torch.no_grad():
      lstm = QuantLSTM(input_size, hidden_size, num_layers)
      x = torch.randn(seq_length, batch_size, input_size) 
      hx = torch.randn(num_layers, batch_size, hidden_size)
      cx = torch.randn(num_layers, batch_size, hidden_size)
      y, (hy, cy) = lstm(x, (hx, cx), mode='calibrate')
      y, (hy, cy) = lstm(x, (hx, cx), mode='quantize')
    return y, (hy, cy)


if __name__ == "__main__":
    bs = 128
    # y1 = test_quant_linear(
        # in_features=1344,
        # out_features=512,
        # batch_size=bs
    # )
    # print(y1.shape)
    # print(y1)

    # print('----------')

    # y2 = test_quant_linear(
        # in_features=512,
        # out_features=29,
        # batch_size=bs
    # )
    # print(y2.shape)
    # print(y2)

    seq_len = 10
    test_quant_lstm(
        input_size=240,
        hidden_size=4096,
        num_layers=2,
        batch_size=bs,
        seq_length=seq_len
    )


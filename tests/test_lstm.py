import numpy as np
import torch
import unittest
import rnn
import quant_rnn
import pytorch_quantization.nn as quant_nn

BF_ABS_TOL = 8e-3
BF_REL_TOL = 5e-3

def gen_tensor(shape, precision, value=None):
    if not value:
        if precision == torch.int8:
            return torch.randint(-127, 127, shape, dtype=precision)
        elif precision == torch.uint8:
            return torch.randint(0, 255, shape, dtype=precision)
        else:
            return torch.randn(shape, dtype=precision)
    else:
        return torch.full(shape, value, dtype=precision)


class TestLSTM(unittest.TestCase):
    def __init__(self, input_size, hidden_size, num_layers,
            batch_size, seq_length, precision=torch.float32,
            num_gates=4, num_directions=1):
        self.G = num_gates
        self.D = num_directions
        self.L = num_layers
        self.IC = input_size
        self.OC = hidden_size
        self.N = batch_size
        self.T = seq_length
        self.precision = precision

    def init_inputs(self):
        self.input = gen_tensor((self.T, self.N, self.IC), self.precision)
        self.hx = gen_tensor((self.L*self.D, self.N, self.OC), self.precision)
        self.cx = gen_tensor((self.L*self.D, self.N, self.OC), self.precision)

    def init_weights(self):
        self.weights = {}
        for layer in range(self.L):
            if layer == 0:
                self.weights["weight_ih_l{}".format(layer)] = gen_tensor((self.G*self.OC, self.IC), self.precision)
            else:
                self.weights["weight_ih_l{}".format(layer)] = gen_tensor((self.G*self.OC, self.D*self.OC), self.precision)
            self.weights["weight_hh_l{}".format(layer)] = gen_tensor((self.G*self.OC, self.OC), self.precision)
            self.weights["bias_ih_l{}".format(layer)] = gen_tensor((self.G*self.OC,), self.precision)
            self.weights["bias_hh_l{}".format(layer)] = gen_tensor((self.G*self.OC,), self.precision)

    def pt_lstm(self):
        with torch.no_grad():
            lstm = torch.nn.LSTM(self.IC, self.OC, self.L)
            for key, value in self.weights.items():
                setattr(lstm, key, torch.nn.Parameter(value, requires_grad=False))
            y, (hy, cy) = lstm(self.input, (self.hx, self.cx))
            self.dump_res(y, hy, cy, msg="pt_lstm:")
        return [y, hy, cy]

    def db_lstm(self):
        lstm = rnn.LSTM(self.IC, self.OC, self.L)
        for key, value in self.weights.items():
            setattr(lstm, key, torch.nn.Parameter(value, requires_grad=False))
        lstm._set_all_weights()
        y, (hy, cy) = lstm(self.input, (self.hx, self.cx))
        self.dump_res(y, hy, cy, msg="db_lstm:")
        return [y, hy, cy]

    def db_quant_lstm(self):
        lstm = quant_rnn.QuantLSTM(self.IC, self.OC, self.L)
        for key, value in self.weights.items():
            setattr(lstm, key, torch.nn.Parameter(value, requires_grad=False))
        lstm._set_all_weights()
        y, (hy, cy) = lstm(self.input, (self.hx, self.cx))
        self.dump_res(y, hy, cy, msg="db_quant_lstm:")
        return [y, hy, cy]

    def trt_lstm(self):
        lstm = quant_nn.QuantLSTM(self.IC, self.OC, self.L)
        for key, value in self.weights.items():
            setattr(lstm, key, torch.nn.Parameter(value, requires_grad=False))
        y, (hy, cy) = lstm(self.input, (self.hx, self.cx))
        self.dump_res(y, hy, cy, msg="trt_lstm:")
        return [y, hy, cy]

    def cmp_res(self, res_a, res_b):
        self.assertTrue(len(res_a), len(res_b))
        for i in range(len(res_a)):
            self.assertTrue(res_a[i].shape, res_b[i].shape)
            self.assertTrue(np.all(np.isclose(res_a[i], res_b[i].cpu(), rtol=BF_REL_TOL, atol=BF_ABS_TOL)))

    def dump_res(self, y, hy, cy, msg=""):
        print(f"{msg}y,{y.shape},{y.dtype},{y.device},sum:{y.sum():.5f},max:{y.max():.5f}")
        #  print(y)
        print(f"{msg}hy,{hy.shape},{hy.dtype},{hy.device},sum:{hy.sum():.5f},max:{hy.max():.5f}")
        #  print(hy)
        print(f"{msg}cy,{cy.shape},{cy.dtype},{cy.device},sum:{cy.sum():.5f},max:{cy.max():.5f}")
        #  print(cy)

    def test_and_cmp(self):
        res_pt = self.pt_lstm()
        print("-"*50)
        res_db = self.db_lstm()
        print("-"*50)
        res_trt = self.trt_lstm()
        print("-"*50)
        res_db_quant = self.db_quant_lstm()
        self.cmp_res(res_pt, res_db)
        #  self.cmp_res(res_pt, res_db_quant)
        self.cmp_res(res_trt, res_db_quant)


if __name__ == '__main__':
     print('========== Test LSTM ==========')
     test_lstm = TestLSTM(
         input_size=5, hidden_size=20, num_layers=1,
         batch_size=3, seq_length=2, precision=torch.float32)
     test_lstm.init_inputs()
     test_lstm.init_weights()
     test_lstm.test_and_cmp()

import os
import torch
import _C as P

from quant_modules import *
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple


class QuantLSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, skip_quant_y, **kwargs):
        super(QuantLSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)
        self.skip_quant_y = skip_quant_y
        self.weights = []
        self.o_scale_list = []
        self.input_scale_list = []
        self.output_scale_list = []

    def _init_layers(self, run_mode=None):
        input_size = self.input_size
        for layer in range(self.num_layers):
            # set quant desc
            if run_mode == "calib":
                QUANT_LSTM_ACTIVA.mode = "calib"
                QUANT_LSTM_WEIGHT.mode = "quant"
            else:
                QUANT_LSTM_ACTIVA.mode = run_mode
                QUANT_LSTM_WEIGHT.mode = run_mode
            # create lstm layer
            if run_mode == "quant":
                lstm_layer = iLSTMLayer(input_size, self.hidden_size)
            else:
                lstm_layer = QuantLSTMLayer(input_size, self.hidden_size)

            if run_mode != None and run_mode != "f32":
                lstm_layer._init_quantizers(
                   WeightQuantizer(QUANT_LSTM_WEIGHT),
                   TensorQuantizer(QUANT_LSTM_ACTIVA),
                   TensorQuantizer(QUANT_LSTM_ACTIVA))

            lstm_layer._init_parameters(
                self.all_weights[layer][0],
                self.all_weights[layer][1],
                self.all_weights[layer][2],
                self.all_weights[layer][3])

            setattr(self, f"lstm{layer}", lstm_layer)
            input_size = self.hidden_size

    def _process_parameters(self, run_mode):
        for layer in range(self.num_layers):
            if run_mode == "quant" or run_mode == "fake_quant":
                getattr(self, f"lstm{layer}")._quant_parameters(self.weights, self.o_scale_list)
            delattr(self, self._all_weights[layer][0])
            delattr(self, self._all_weights[layer][1])
            delattr(self, self._all_weights[layer][2])
            delattr(self, self._all_weights[layer][3])

    def _propagate_quantizers(self):
        # per-module
        # first_cell = getattr(self, "lstm0")
        # for layer in range(self.num_layers):
            # cur_cell = getattr(self, f"lstm{layer}")
            # cur_cell.input_quantizer = first_cell.input_quantizer
            # cur_cell.output_quantizer = first_cell.input_quantizer

        # per-layer
        for layer in range(self.num_layers):
            if layer != self.num_layers - 1:
                cur_cell = getattr(self, f"lstm{layer}")
                next_cell = getattr(self, f"lstm{layer+1}")
                cur_cell.output_quantizer = next_cell.input_quantizer
            self.input_scale_list.append(getattr(self, f"lstm{layer}").input_quantizer.scale.item())
            self.output_scale_list.append(getattr(self, f"lstm{layer}").output_quantizer.scale.item())


    @torch.jit.ignore
    def forward(self, x: Tensor, state: Optional[Tuple[List[Tensor], List[Tensor]]]=None) -> Tuple[Tensor, Tuple[List[Tensor], List[Tensor]]]:
        hx, cx = [], []
        if state is None:
            for i in range(self.num_layers):
                hx.append(torch.zeros(x.size(1), self.hidden_size,
                        dtype=x.dtype))
                cx.append(torch.zeros(x.size(1), self.hidden_size,
                        dtype=torch.float16))
        else:
            (hx, cx) = state[:]

        for layer in range(self.num_layers):
            x, (hx[layer], cx[layer]) = getattr(self, f"lstm{layer}")(x, hx[layer], cx[layer], layer==(self.num_layers-1) and self.skip_quant_y)
        return (x, (hx, cx))

class QuantLSTMLayer(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(QuantLSTMLayer, self).__init__(input_size, hidden_size, **kwargs)

    def _init_quantizers(self,
            weight_quantizer: WeightQuantizer=None,
            input_quantizer: TensorQuantizer=None,
            output_quantizer: TensorQuantizer=None) -> None:
        self.weight_quantizer = weight_quantizer
        self.input_quantizer = input_quantizer
        self.output_quantizer = output_quantizer

    def _init_parameters(self,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor, b_hh: Tensor) -> None:
        self.weight_ih = w_ih
        self.weight_hh = w_hh
        self.bias_ih = b_ih
        self.bias_hh = b_hh

    def _quant_parameters(self, weights, o_scale_list) -> None:
        self.weight_quantizer.amax = torch.max(
            torch.cat([self.weight_ih, self.weight_hh], 1).abs())
        self.weight_ih = Parameter(
            self.weight_quantizer(self.weight_ih), requires_grad=False)
        self.weight_hh = Parameter(
            self.weight_quantizer(self.weight_hh), requires_grad=False)

    def forward(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor, quant_y: bool=False) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hasattr(self, "input_quantizer"):
            if self.input_quantizer.mode != None:
                xt, ht_1 = self.input_quantizer(
                    torch.cat([xt, ht_1], 1)).split([xt.size(1), ht_1.size(1)], 1)
        gates_list = []
        for i in range(xt.shape[0]):
            gates = F.linear(xt[i], self.weight_ih, self.bias_ih)
            gates_list.append(gates)

        yt_list = []
        for i in range(xt.shape[0]):
            gates_list[i] += F.linear(ht_1, self.weight_hh, self.bias_hh)
            it, ft, gt, ot = gates_list[i].chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct_1 = (ft * ct_1) + (it * gt)
            ht_1 = ot * torch.tanh(ct_1)
            yt_list.append(ht_1)
        yt = torch.stack(yt_list, 0)
        return yt, (ht_1, ct_1)


class iLSTMLayer(QuantLSTMLayer):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(iLSTMLayer, self).__init__(input_size, hidden_size, **kwargs)

    def _quant_parameters(self, weights, o_scale_list) -> None:
        self.weight_quantizer.amax = torch.max(
            torch.cat([self.weight_ih, self.weight_hh], 1).abs())
        self.weight_ih = Parameter(
            self.weight_quantizer._quant_forward(self.weight_ih), requires_grad=False)
        self.weight_hh = Parameter(
            self.weight_quantizer._quant_forward(self.weight_hh), requires_grad=False)
        b_scale = self.input_quantizer.scale * self.weight_quantizer.scale
        self.bias_ih = Parameter(self.bias_ih * b_scale, requires_grad=False)
        self.bias_hh = Parameter(self.bias_hh * b_scale, requires_grad=False)
        self.o_scale = 1 / b_scale.item()
        self.in_quant_scale = self.input_quantizer.scale.item()
        # self.out_quant_scale = self.output_quantizer.scale.item()
        weights_layer = [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        weights.append(weights_layer)
        o_scale_list.append(self.o_scale)

    def forward(self, x: Tensor, ht_1: Tensor, ct_1: Tensor, skip_quant_y: bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        gates_list = []
        for i in range(x.shape[0]):
            gates = P.linear(x[i].squeeze(0), self.weight_ih, self.bias_ih, self.o_scale, None)
            gates_list.append(gates)

        yt_list = []
        for i in range(x.shape[0]):
            gates_list[i] += P.linear(ht_1, self.weight_hh, self.bias_hh, self.o_scale, None)

            it, ft, gt, ot = gates_list[i].chunk(4, 1)
            y_p = P.lstm_postop(it, ft, gt, ot, ct_1,
                self.in_quant_scale, self.output_quantizer.scale.item(), skip_quant_y)
            
            ht_1 = y_p[2]
            ct_1 = y_p[3]
            if skip_quant_y:
                yt_list.append(y_p[0])
            else:
                yt_list.append(y_p[1])

        x = torch.stack(yt_list, 0)
        return (x, (ht_1, ct_1))
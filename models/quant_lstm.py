import os
import torch
import _C as P

from quant_modules import *
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple


class QuantLSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(QuantLSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)

    def _init_cells(self, run_mode=None):
        input_size = self.input_size
        for layer in range(self.num_layers):
            # set quant desc
            if run_mode == "calib":
                QUANT_LSTM_ACTIVA.mode = "calib"
                QUANT_LSTM_WEIGHT.mode = "quant"
            else:
                QUANT_LSTM_ACTIVA.mode = run_mode
                QUANT_LSTM_WEIGHT.mode = run_mode
            # create lstm cell
            if run_mode == "quant":
                lstm_cell = iLSTMCell(input_size, self.hidden_size)
            else:
                lstm_cell = QuantLSTMCell(input_size, self.hidden_size)

            if run_mode != None and run_mode != "f32":
                lstm_cell._init_quantizers(
                   WeightQuantizer(QUANT_LSTM_WEIGHT),
                   TensorQuantizer(QUANT_LSTM_ACTIVA),
                   TensorQuantizer(QUANT_LSTM_ACTIVA))

            lstm_cell._init_parameters(
                self.all_weights[layer][0],
                self.all_weights[layer][1],
                self.all_weights[layer][2],
                self.all_weights[layer][3])

            setattr(self, f"lstm{layer}", lstm_cell)
            input_size = self.hidden_size

    def _process_parameters(self, run_mode):
        for layer in range(self.num_layers):
            if run_mode == "quant" or run_mode == "fake_quant":
                getattr(self, f"lstm{layer}")._quant_parameters()
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

        # pre-layer
        for layer in range(self.num_layers):
            if layer != self.num_layers - 1:
                cur_cell = getattr(self, f"lstm{layer}")
                next_cell = getattr(self, f"lstm{layer+1}")
                cur_cell.output_quantizer = next_cell.input_quantizer


    @torch.jit.ignore
    def forward(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None:
            hx = torch.zeros(self.num_layers, x.size(1), self.hidden_size,
                    dtype=x.dtype)
            cx = torch.zeros(self.num_layers, x.size(1), self.hidden_size,
                    dtype=torch.float32)
        else:
            (hx, cx) = state[:]

        x = x.contiguous()
        hy_list, cy_list = [], []
        for layer in range(self.num_layers):
            x, (hy_layer, cy_layer) = self.lstm_layer(x, hx[layer], cx[layer], layer)
            hy_list.append(hy_layer)
            cy_list.append(cy_layer)
        hy = torch.stack(hy_list, 0)
        cy = torch.stack(cy_list, 0)
        return x, (hy, cy)

    @torch.jit.ignore
    def lstm_layer(self, x: Tensor, hx: Tensor, cx: Tensor, layer: int) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        y_list = []
        for step in range(x.size(0)):
            cell_y, hx, cx = getattr(self, f"lstm{layer}")(
                x[step], hx, cx, layer==(self.num_layers-1))
            y_list.append(cell_y)
        y = torch.stack(y_list, 0)
        return y, (hx, cx)


class QuantLSTMCell(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(QuantLSTMCell, self).__init__(input_size, hidden_size, **kwargs)

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

    def _quant_parameters(self) -> None:
        self.weight_quantizer.amax = torch.max(
                torch.cat([self.weight_ih, self.weight_hh], 1).abs())
        self.weight_quantizer.scale = self.weight_quantizer._max_bound / self.weight_quantizer.amax
        self.weight_ih = Parameter(
            self.weight_quantizer(self.weight_ih), requires_grad=False)
        self.weight_hh = Parameter(
            self.weight_quantizer(self.weight_hh), requires_grad=False)

    def forward(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor, last_layer: bool=False) -> List[Tensor]:
        if hasattr(self, "input_quantizer"):
            if self.input_quantizer.mode != None:
                xt, ht_1 = self.input_quantizer(
                    torch.cat([xt, ht_1], 1)).split([xt.size(1), ht_1.size(1)], 1)
        
        gates = F.linear(xt, self.weight_ih, self.bias_ih)
        gates += F.linear(ht_1, self.weight_hh, self.bias_hh)
        it, ft, gt, ot = gates.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        ht = ot * torch.tanh(ct)
        return ht, ht, ct


class iLSTMCell(QuantLSTMCell):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(iLSTMCell, self).__init__(input_size, hidden_size, **kwargs)

    def _quant_parameters(self) -> None:
        self.weight_quantizer.amax = torch.max(
                torch.cat([self.weight_ih, self.weight_hh], 1).abs())
        self.weight_quantizer.scale = self.weight_quantizer._max_bound / self.weight_quantizer.amax
        self.weight_ih = Parameter(
            self.weight_quantizer._quant_forward(self.weight_ih), requires_grad=False)
        self.weight_hh = Parameter(
            self.weight_quantizer._quant_forward(self.weight_hh), requires_grad=False)
        b_scale = self.input_quantizer.scale * self.weight_quantizer.scale
        self.bias_ih = Parameter(self.bias_ih * b_scale, requires_grad=False)
        self.bias_hh = Parameter(self.bias_hh * b_scale, requires_grad=False)
        self.o_scale = 1 / b_scale

    def forward(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor, last_layer: bool) -> List[Tensor]:
        gates = P.linear(xt, self.weight_ih, self.bias_ih, self.o_scale.item(), None)
        # TODO: linear_add fusion
        gates += P.linear(ht_1, self.weight_hh, self.bias_hh, self.o_scale.item(), None)
        it, ft, gt, ot = gates.chunk(4, 1)
        # TODO: approximate + fp16
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        # TODO: requantize before output
        ht = ot * torch.tanh(ct)
        # TODO: quant last layer
        if last_layer:
            yt = ht.clone()
        else:
            yt = self.output_quantizer(ht)
        ht = self.input_quantizer(ht)
        # yt = ht
        return yt, ht, ct


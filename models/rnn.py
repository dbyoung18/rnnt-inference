import os
import numpy as np
import torch
import torch.nn as nn
import _C as P
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple
from quant_modules import *


class LSTM(QuantLSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)

    def _init_weights(self):
        self.weights : List[Tensor] = self.all_weights

    def quant_params(self, layer: int,
            input_quantizer: TensorQuantizer, weight_quantizer: WeightQuantizer) -> List[Tensor]:
        w_ih, w_hh, b_ih, b_hh = self.weights[layer][:]
        weight_quantizer.amax = torch.max(torch.cat([w_ih, w_hh], 1).abs())
        weight_quantizer.scale = weight_quantizer._max_bound / weight_quantizer.amax
        b_scale = input_quantizer.scale * weight_quantizer.scale
        o_scale = 1 / b_scale
        self.weights[layer][0] = weight_quantizer._quant_forward(w_ih)
        self.weights[layer][1] = weight_quantizer._quant_forward(w_hh)
        self.weights[layer][2] = b_ih*b_scale
        self.weights[layer][3] = b_hh*b_scale
        return self.weights[layer][0], self.weights[layer][1], self.weights[layer][2], self.weights[layer][3], o_scale

    def forward(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None):
        if state is None:
            hx = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype)
            cx = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=torch.float32)
        else:
            (hx, cx) = state[:]

        hy_list, cy_list = [], []
        for layer in range(self.num_layers):
            input_quantizer : TensorQuantizer = self._input_quantizers[layer]
            weight_quantizer : WeightQuantizer = self._weight_quantizers[layer]
            layer_hx, layer_cx = hx[layer], cx[layer]
            w_ih, w_hh, b_ih, b_hh, o_scale = self.quant_params(layer, input_quantizer, weight_quantizer)
            x, (layer_hx, layer_cx) = self.lstm_layer(
                x, layer_hx, layer_cx,
                w_ih, w_hh, b_ih, b_hh,
                o_scale, layer)
            hy_list.append(layer_hx)
            cy_list.append(layer_cx)

        hy = torch.stack(hy_list, 0)
        cy = torch.stack(cy_list, 0)
        return x, (hy, cy)

    def lstm_layer(self, x: Tensor, hx: Tensor, cx: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None,
            o_scale=None, layer: int=0):
        y_list = []
        for step in range(x.size(0)):
            y, hx, cx = self.lstm_cell(
                x[step], hx, cx,
                w_ih, w_hh, b_ih, b_hh,
                o_scale)
            input_quantizer: TensorQuantizer = self._input_quantizers[layer]
            hx = input_quantizer._quant_forward(hx)
            y_list.append(y)
        y = torch.stack(y_list, 0)
        if layer != self.num_layers - 1:  # TODO: quant laster layer
            next_quantizer: TensorQuantizer = self._input_quantizers[layer+1]
            y = next_quantizer._quant_forward(y)
        return y, (hx, cx)

    def lstm_cell(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None,
            o_scale: Tensor=None) -> List[Tensor]:
        """
        Args:
            xt: {T, N, IC} -> {N, IC}
            ht_1: {D*L, N, OC} -> {N, OC}
            ct_1: {D*L, N, OC} -> {N, OC}
            w_ih: {G*OC, IC} -> {4OC, IC}
            w_hh: {G*OC, OC} -> {4OC, OC}
            b_ih: {G*OC} -> {4OC}
            b_hh: {G*OC} -> {4OC}

        Returns:
            yt = ht: {N, OC}
            ct: {N, OC}
        """
        gates = P.linear(xt, w_ih, b_ih, o_scale.item(), None)
        # TODO: linear_add fusion
        gates += P.linear(ht_1, w_hh, b_hh, o_scale.item(), None)
        it, ft, gt, ot = gates.chunk(4, 1)
        # TODO: approximate + fp16
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        ht = ot * torch.tanh(ct)
        return ht, ht, ct


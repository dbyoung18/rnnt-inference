import os
import numpy as np
import torch
import torch.nn as nn
import _C as P
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import List, Optional, Tuple
from quant_modules import *


class LSTM(QuantLSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)

    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None):
        if state is None:
            hx = cx = torch.zeros(self.num_layers, input.size(1), self.hidden_size,
                                  dtype=input.dtype, device=input.device, requires_grad=False)
        else:
            (hx, cx) = state[:]

        if input.dtype == torch.float32:
            layer_x = self._input_quantizers[0](input.contiguous())
            hx = self._input_quantizers[0](hx.contiguous())
        hy_list, cy_list = [], []
        for layer in range(self.num_layers):
            layer_hx, layer_cx = hx[layer], cx[layer]
            input_quantizer = self._input_quantizers[layer]
            weight_quantizer = self._weight_quantizers[layer]
            weight_quantizer.amax = torch.max(
                torch.cat([self.all_weights[layer][0], self.all_weights[layer][1]], 1).abs())
            weight_quantizer.scale = weight_quantizer._max_bound / weight_quantizer.amax
            weight_quantizer._mode = "quant"
            b_scale = input_quantizer.scale * weight_quantizer.scale
            o_scale = 1 / b_scale
            setattr(self, f"weight_ih_l{layer}",
                Parameter(weight_quantizer(self.all_weights[layer][0]), requires_grad=False))
            setattr(self, f"weight_hh_l{layer}",
                Parameter(weight_quantizer(self.all_weights[layer][1]), requires_grad=False))
            setattr(self, f"bias_ih_l{layer}",
                Parameter(self.all_weights[layer][2]*b_scale, requires_grad=False))
            setattr(self, f"bias_hh_l{layer}",
                Parameter(self.all_weights[layer][3]*b_scale, requires_grad=False))

            layer_x, (layer_hx, layer_cx) = self.lstm_layer(
                layer_x, layer_hx, layer_cx,
                self.all_weights[layer][0], self.all_weights[layer][1],
                self.all_weights[layer][2], self.all_weights[layer][3],
                o_scale, layer)

            hy_list.append(layer_hx)
            cy_list.append(layer_cx)

        y = layer_x
        hy = torch.stack(hy_list, 0)
        cy = torch.stack(cy_list, 0)
        return y, (hy, cy)

    def lstm_layer(self, x: Tensor, hx: Tensor, cx: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None,
            o_scale=None, layer=0):
        y_list = []
        for step in range(x.size(0)):
            y, hx, cx = self.lstm_cell(
                x[step], hx, cx,
                w_ih, w_hh, b_ih, b_hh,
                o_scale)
            hx = self._input_quantizers[layer](hx)
            if layer != self.num_layers - 1:
                y = self._input_quantizers[layer+1](y)
            y_list.append(y)
        y = torch.stack(y_list, 0)
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


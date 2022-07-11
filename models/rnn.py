import os
import numpy as np
import torch
import torch.nn as nn
import plugins as P
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import List, Optional, Tuple
from quant_rnn import TensorQuantizer


class LSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers)
        self._set_all_weights()
        self.run_mode = kwargs.pop("run_mode", None)
        if self.run_mode is not None:
            self._init_quantizers()

    def _init_quantizers(self):
        QUANT_LSTM_INPUT.mode = self.run_mode
        self._input_quantizers = nn.ModuleList(
            [TensorQuantizer(QUANT_LSTM_INPUT) for _ in range(self.num_layers)])
        if self.run_mode != "calib":
            self._weight_quantizers = nn.ModuleList(
                [TensorQuantizer(QUANT_LSTM_WEIGHT) for _ in range(self.num_layers)])

    def _set_all_weights(self, weights=None) -> List[List[Parameter]]:
        if weights == None:
            setattr(self, "weights",
                [[getattr(self, weight) for weight in weights] for weights in self._all_weights])
        else:
            setattr(self, "weights", weights)

    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]]=None):
        if state is None:
            hx = cx = torch.zeros(self.num_layers, input.size(1), self.hidden_size,
                                  dtype=input.dtype, device=input.device, requires_grad=False)
        else:
            (hx, cx) = state[:]

        layer_x = input.contiguous()
        hy_list, cy_list = [], []
        for layer in range(self.num_layers):
            layer_hx, layer_cx = hx[layer], cx[layer]
            (w_ih, w_hh, b_ih, b_hh) = self.weights[layer][:]
            input_quantizer = self._input_quantizers[layer] if hasattr(self, "_input_quantizers") else None
            weight_quantizer = self._weight_quantizers[layer] if hasattr(self, "_weight_quantizers") else None

            layer_x, (layer_hx, layer_cx) = self.lstm_layer(
                layer_x, layer_hx, layer_cx,
                w_ih, w_hh, b_ih, b_hh,
                input_quantizer, weight_quantizer)

            hy_list.append(layer_hx)
            cy_list.append(layer_cx)

        y = layer_x
        hy = torch.stack(hy_list, 0)
        cy = torch.stack(cy_list, 0)
        return y, (hy, cy)

    def lstm_layer(self, x: Tensor, hx: Tensor, cx: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None,
            input_quantizer=None, weight_quantizer=None):
        y_list = []
        for step in range(x.size(0)):
            hx, cx = self.lstm_cell(
                x[step], hx, cx,
                w_ih, w_hh, b_ih, b_hh,
                input_quantizer, weight_quantizer)
            y_list.append(hx)
        y = torch.stack(y_list, 0)
        return y, (hx, cx)

    def lstm_cell(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None,
            input_quantizer=None, weight_quantizer=None) -> List[Tensor]:
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
        if input_quantizer != None:
            xt, ht_1 = input_quantizer(torch.cat([xt, ht_1], 1)).split([xt.size(1), ht_1.size(1)], 1)
        if weight_quantizer != None:
            w_ih, w_hh = weight_quantizer(torch.cat([w_ih, w_hh], 1)).split([w_ih.size(1), w_hh.size(1)], 1)

        if self.run_mode == "quant":
            o_scale = input_quantizer.scale * weight_quantizer.scale
            gates = P.linear(xt, w_ih, b_ih, o_scale, None) + P.linear(ht_1, w_hh, b_hh, o_scale, None) 
            #  quant_b = (b_ih + b_hh) * o_scale
            #  gates = torch.matmul(xt, torch.t(w_ih)) + torch.matmul(ht_1, torch.t(w_hh)) + quant_b
            #  gates /= o_scale
        else:
            gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
        it, ft, gt, ot = gates.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        ht = ot * torch.tanh(ct)
        return ht, ct


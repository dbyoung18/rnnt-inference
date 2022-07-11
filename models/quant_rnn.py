import os
import numpy as np
import torch
import torch.nn as nn
import _C as P
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import List, Optional, Tuple


def round_and_clamp(input, _min: float, _max: float):
    return torch.clamp(input.round(), _min, _max)


def clamp_and_round(input, _min: float, _max: float):
    return torch.round(torch.clamp(input, _min, _max))


class QuantDescriptor():
    def __init__(self, axis=None, amax=None, mode="quant", **kwargs):
        self._axis = axis
        self._amax = amax
        self._mode = mode
        self._update_amax = kwargs.pop("update_amax", False)
        self._narrow_bound = kwargs.pop("narrow_bound", False)

    @property
    def axis(self):
        return self._axis

    @property
    def amax(self):
        return self._amax

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def update_amax(self):
        return self._update_amax

    @property
    def narrow_bound(self):
        return self._narrow_bound


# per-tensor & per-cell
QUANT_LSTM_INPUT = QuantDescriptor(axis=None, amax=None, mode="fake_quant", update_amax=False)
# per-oc & per-layer
QUANT_LSTM_WEIGHT = QuantDescriptor(axis=None, amax=None, mode="fake_quant")


class Calibrator():
    def __init__(self, axis=None, track_amax=True):
        self._axis = axis
        self._calib_amax = None
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []

    def collect(self, x):
        # TODO: update amax per axis
        cur_amax = x.abs().max()
        self._calib_amax = cur_amax if self._calib_amax is None else torch.max(self._calib_amax, cur_amax)
        if self._track_amax:
            self._amaxs.append(cur_amax)
        return self._calib_amax

    def reset(self):
        self._calib_amax = None

    def get_amax(self):
        return self._calib_amax


class TensorQuantizer(nn.Module):
    """
    Tensor Quantizer contain same buffers as pytorch-quantization

    Args:
        mode: calib/quant/fake_quant/dynamic_quant

    """
    def __init__(self, quant_desc=QuantDescriptor(), **kwargs):
        super(TensorQuantizer, self).__init__()
        self._mode = quant_desc.mode
        self._axis = quant_desc._axis
        if quant_desc._amax is not None:
            self.register_buffer("_amax", torch.tensor(quant_desc._amax))
        #  else:
            #  self.register_buffer("_amax", None)
        self._scale = None
        self._calibrator = None if quant_desc.mode != "calib" else Calibrator(self._axis)
        self._bound = torch.tensor(127., dtype=torch.float32)
        self._name = kwargs.pop("name", None)
        self._update_amax = quant_desc.update_amax
        self._narrow_bound = quant_desc.narrow_bound

    @property
    def amax(self):
        return self._amax if hasattr(self, "_amax") else None

    @amax.setter
    def amax(self, value):
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", value)
        else:
            self._amax = value

    @property
    def scale(self):
        return self._scale if self._scale is not None else self._bound / self._amax

    @scale.setter
    def scale(self, value):
        self._scale = value

    def _quant_forward(self, inputs):
        if self._scale is None or self._update_amax:
            self.amax = torch.max(inputs.abs(), self._axis).values.unsqueeze(self._axis) \
                if self._axis is not None else torch.max(inputs.abs())
        
        min_bound = -self._bound if self._narrow_bound else -self._bound - 1
        outputs = round_and_clamp(inputs * self.scale, min_bound, self._bound)
        if self._mode == "fake_quant":
            outputs /= self.scale
        return outputs

    def forward(self, inputs):
        outputs = inputs
        if self._mode == "calib":
            self.amax = self._calibrator.collect(inputs)
        elif self._mode == "quant" or self._mode == "fake_quant":
            outputs = self._quant_forward(inputs)
        return outputs

    def __str__(self):
        s = (self._name + ': ') if self._name is not None else 'TensorQuantizer'
        s += f" mode=${self._mode}"
        s += f" axis=${self._axis}"
        s += f" amax=${self._amax}"
        s += f" scale=${self._scale}"
        s += f" bound=${self._bound}"
        return s


class QuantLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(QuantLSTM, self).__init__(input_size, hidden_size, num_layers)
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
            xt = xt.type(torch.int8)
            ht_1 = ht_1.type(torch.int8)
            w_ih = w_ih.type(torch.int8)
            w_hh = w_hh.type(torch.int8)
            gates = P.linear(xt, w_ih, b_ih, o_scale.item(), None) + P.linear(ht_1, w_hh, b_hh, o_scale.item(), None) 
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

        ft = round_and_clamp(ft*127, -127, 127) / 127
        ct_1 = round_and_clamp(ct_1*127, -127, 127) / 127
        it = round_and_clamp(it*127, -127, 127) / 127
        gt = round_and_clamp(gt*127, -127, 127) / 127

        ct = (ft * ct_1) + (it * gt)

        ht = ot * torch.tanh(ct)
        return ht, ct


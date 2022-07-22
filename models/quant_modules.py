import os
import numpy as np
import torch
import torch.nn as nn
import _C as P
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import List, Optional, Tuple
import approximate


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
        mode: calib/quant/fake_quant

    """
    def __init__(self, quant_desc=QuantDescriptor(), **kwargs):
        super(TensorQuantizer, self).__init__()
        self._mode = quant_desc.mode
        self._axis = quant_desc._axis
        self._max_bound = torch.tensor(127., dtype=torch.float32)
        self._min_bound = -self._max_bound if quant_desc.narrow_bound else -self._max_bound-1
        self.amax = torch.tensor(quant_desc._amax) if quant_desc._amax != None else torch.tensor(0.)
        self._scale = None if self.amax == torch.tensor(0.) else self._max_bound / self.amax
        self._name = kwargs.pop("name", None)
        self._update_amax = quant_desc.update_amax  # dynamic

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
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def _quant_forward(self, inputs):
        self.scale = self._max_bound / self.amax
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs = outputs.type(torch.int8)
        return outputs

    @torch.jit.ignore
    def _fake_quant_forward(self, inputs):
        if self._update_amax:
            if self._axis != None:
                self.amax = torch.max(inputs.abs(), self._axis).values.unsqueeze(self._axis)
            else:
                self.amax = torch.max(inputs.abs())

        self.scale = self._max_bound / self.amax
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs /= self.scale
        return outputs

    def forward(self, inputs):
        outputs = inputs
        if self._mode == "calib":
            self.amax = torch.max(self.amax, inputs.abs().max())
        elif self._mode == "quant":
            outputs = self._quant_forward(inputs)
        elif self._mode == "fake_quant":
            outputs = self._fake_quant_forward(inputs)
        return outputs

    def __str__(self):
        s = (self._name + ': ') if self._name != None else 'TensorQuantizer'
        s += f" mode=${self._mode}"
        s += f" axis=${self._axis}"
        s += f" amax=${self._amax}"
        s += f" scale=${self._scale}"
        s += f" max_bound=${self._max_bound}"
        s += f" min_bound=${self._min_bound}"
        return s


class WeightQuantizer(TensorQuantizer):
    def __init__(self, quant_desc=QuantDescriptor(), **kwargs):
        super(WeightQuantizer, self).__init__(quant_desc)

    def _quant_forward(self, inputs: Tensor) -> Tensor:
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs = outputs.type(torch.int8)
        prepacked_outputs = P.prepack_linear_weight(outputs)
        return prepacked_outputs

    @torch.jit.ignore
    def _fake_quant_forward(self, inputs):
        if self._scale == None:
            if self._axis != None:
                self.amax = torch.max(inputs.abs(), self._axis).values.unsqueeze(self._axis)
            else:
                self.amax = torch.max(inputs.abs())
            self.scale = self._max_bound / self.amax
        
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs /= self.scale
        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        if self._mode == "quant":
            outputs = self._quant_forward(inputs)
        elif self._mode == "fake_quant":
            outputs = self._fake_quant_forward(inputs)
        return outputs


class QuantLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(QuantLSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)

    def _init_quantizers(self, run_mode=None):
        if run_mode == 'calib':
            QUANT_LSTM_INPUT.mode = 'calib'
            QUANT_LSTM_WEIGHT.mode = 'quant'
        else:
            QUANT_LSTM_INPUT.mode = run_mode
            QUANT_LSTM_WEIGHT.mode = run_mode
        self._input_quantizers : nn.ModuleList = nn.ModuleList(
            [TensorQuantizer(QUANT_LSTM_INPUT) for _ in range(self.num_layers)])
        self._weight_quantizers : nn.ModuleList = nn.ModuleList(
            [WeightQuantizer(QUANT_LSTM_WEIGHT) for _ in range(self.num_layers)])

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
            (w_ih, w_hh, b_ih, b_hh) = self.all_weights[layer][:]
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
        if weight_quantizer != None and weight_quantizer._mode != None and input_quantizer._mode != "calib": 
            w_ih, w_hh = weight_quantizer(torch.cat([w_ih, w_hh], 1)).split([w_ih.size(1), w_hh.size(1)], 1)
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
        if input_quantizer != None and input_quantizer._mode != None:
            xt, ht_1 = input_quantizer(torch.cat([xt, ht_1], 1)).split([xt.size(1), ht_1.size(1)], 1)

        gates = F.linear(xt, w_ih, b_ih)
        gates += F.linear(ht_1, w_hh, b_hh)
        it, ft, gt, ot = gates.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        ht = ot * torch.tanh(ct)
        return ht, ct


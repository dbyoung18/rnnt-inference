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
QUANT_LSTM_ACTIVA = QuantDescriptor(axis=None, amax=None, mode="fake_quant", update_amax=False)
# per-oc & per-layer
QUANT_LSTM_WEIGHT = QuantDescriptor(axis=None, amax=None, mode="fake_quant")


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
        self._scale = torch.tensor(0.)
        self._name = kwargs.pop("name", "TensorQuantizer")
        self._update_amax = quant_desc.update_amax  # dynamic

    @property
    def mode(self):
        return self._mode

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
        return self._max_bound / self.amax

    @scale.setter
    def scale(self, value):
        self._scale = value

    @torch.jit.export
    def _quant_forward(self, inputs: Tensor) -> Tensor:
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

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        if self._mode == "calib":
            self.amax = torch.max(self.amax, inputs.abs().max())
        elif self._mode == "quant":
            outputs = self._quant_forward(inputs)
        elif self._mode == "fake_quant":
            outputs = self._fake_quant_forward(inputs)
        return outputs

    def __str__(self):
        s = f" name=${self._name}\n"
        s += f" mode=${self._mode}\n"
        s += f" axis=${self._axis}\n"
        s += f" amax=${self._amax}\n"
        s += f" scale=${self._scale}\n"
        s += f" max_bound=${self._max_bound}\n"
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


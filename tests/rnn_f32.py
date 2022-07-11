import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple


class LSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers)
        self._set_all_weights()

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

            layer_x, (layer_hx, layer_cx) = self.lstm_layer(
                layer_x, layer_hx, layer_cx, w_ih, w_hh, b_ih, b_hh)

            hy_list.append(layer_hx)
            cy_list.append(layer_cx)

        y = layer_x
        hy = torch.stack(hy_list, 0)
        cy = torch.stack(cy_list, 0)
        return y, (hy, cy)

    def lstm_layer(self, x: Tensor, hx: Tensor, cx: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None):
        y_list = []
        for step in range(x.size(0)):
            hx, cx = self.lstm_cell(x[step], hx, cx, w_ih, w_hh, b_ih, b_hh)
            y_list.append(hx)
        y = torch.stack(y_list, 0)
        return y, (hx, cx)

    def lstm_cell(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor,
            w_ih: Tensor, w_hh: Tensor,
            b_ih: Tensor=None, b_hh: Tensor=None) -> List[Tensor]:
        """
        Input:
          xt: {T, N, IC} -> {N, IC}
          ht_1: {D*L, N, OC} -> {N, OC}
          ct_1: {D*L, N, OC} -> {N, OC}
          w_ih: {G*OC, IC} -> {4OC, IC}
          w_hh: {G*OC, OC} -> {4OC, OC}
          b_ih: {G*OC} -> {4OC}
          b_hh: {G*OC} -> {4OC}
        Return:
          yt = ht: {N, OC}
          ct: {N, OC}
        """
        gates = F.linear(xt, w_ih, b_ih) + F.linear(ht_1, w_hh, b_hh)
        it, ft, gt, ot = gates.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        ct = (ft * ct_1) + (it * gt)
        ht = ot * torch.tanh(ct)
        return ht, ct


if __name__ == "__main__":
    L = 2
    T = 5
    N = 3
    IC = 10
    OC = 20

    x = torch.randn(T, N, IC)
    hx = torch.randn(L, N, OC)
    cx = torch.randn(L, N, OC)

    lstm = LSTM(IC, OC, L)
    y, (hy, cy) = lstm(x, (hx, cx))
    print(lstm)
    print("done")
 

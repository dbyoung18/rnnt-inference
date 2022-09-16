import torch

from config import RNNTParam
from torch import Tensor
from torch.nn import Linear
from quant_lstm import QuantLSTM as LSTM
from typing import List, Tuple
from utils import *


class RNNT(torch.nn.Module):
    def __init__(self, model_path=None, run_mode=None, load_jit=False):
        super().__init__()
        self.transcription = Transcription(run_mode)
        self.prediction = Prediction()
        self.joint = Joint()
        if model_path is not None:
            saved_quantizers = False if run_mode == None or run_mode == "calib" else True
            self._load_model(model_path, run_mode, load_jit, saved_quantizers)

    def _load_model(self, model_path, run_mode=None, load_jit=False, saved_quantizers=False):
        if load_jit and os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location="cpu")
            self.transcription.pre_quantizer = model.transcription.pre_quantizer
            self.transcription.pre_rnn.lstm0 = model.transcription.pre_rnn.lstm0
            self.transcription.pre_rnn.lstm1 = model.transcription.pre_rnn.lstm1
            self.transcription.stack_time = model.transcription.stack_time
            self.transcription.post_quantizer = model.transcription.post_quantizer
            self.transcription.post_rnn.lstm0 = model.transcription.post_rnn.lstm0
            self.transcription.post_rnn.lstm1 = model.transcription.post_rnn.lstm1
            self.transcription.post_rnn.lstm2 = model.transcription.post_rnn.lstm2
            self.prediction = model.prediction
            self.joint = model.joint
        else:
            model = torch.load(model_path, map_location="cpu")
            state_dict = migrate_state_dict(model)
            if saved_quantizers:
                self.transcription.pre_rnn._init_cells(run_mode)
                self.transcription.post_rnn._init_cells(run_mode)
                # self.joint.linear1_trans._init_quantizers(run_mode)
                self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False)
                self.transcription.pre_rnn._init_cells(run_mode)
                self.transcription.post_rnn._init_cells(run_mode)
                # self.joint.linear1_trans._init_quantizers(run_mode)

            self.transcription.pre_rnn._process_parameters(run_mode)
            self.transcription.post_rnn._process_parameters(run_mode)
            # self.joint.linear1_trans._quant_parameters(run_mode)
            if run_mode == "quant":
                self.transcription.pre_rnn._propagate_quantizers()
                self.transcription.post_rnn._propagate_quantizers()
                self.transcription.pre_rnn.lstm1.output_quantizer = self.transcription.post_rnn.lstm0.input_quantizer
                self.transcription.pre_quantizer = self.transcription.pre_rnn.lstm0.input_quantizer
                self.transcription.post_quantizer = self.transcription.post_rnn.lstm0.input_quantizer


class Transcription(torch.nn.Module):
    def __init__(self, run_mode, **kwargs):
        super().__init__()
        self.pre_rnn = LSTM(
            RNNTParam.trans_input_size,
            RNNTParam.trans_hidden_size,
            RNNTParam.pre_num_layers,
            quant_last_layer=False
        )
        self.stack_time = StackTime(RNNTParam.stack_time_factor)
        self.post_rnn = LSTM(
            RNNTParam.trans_hidden_size*RNNTParam.stack_time_factor,
            RNNTParam.trans_hidden_size,
            RNNTParam.post_num_layers,
            quant_last_layer=True
        )
        # self.run_mode = kwargs.pop("run_mode", None)
        self.run_mode = run_mode

    @torch.jit.ignore
    def forward(self,
            f: Tensor, f_lens: Tensor,
            pre_state: Tuple[Tensor, Tensor]=None,
            post_state: Tuple[Tensor, Tensor]=None) -> Tuple[Tensor, Tensor]:
        """
        Args:
          f: {T, N, IC}
          f_lens: {N}
          pre_state: ({D*L, N, C}, {D*L, N, C})
          post_state: ({D*L, N, C}, {D*L, N, C})

        Returns:
          f: {T, N, OC}
          f_lens: {N}
          pre_state: ({D*L, N, C}, {D*L, N, C})
          post_state: ({D*L, N, C}, {D*L, N, C})
        """
        if self.run_mode == "quant":
            f = self.pre_quantizer(f)
        f, pre_state = self.pre_rnn(f, pre_state)
        f, f_lens = self.stack_time(f, f_lens)
        # if self.run_mode == "quant":
            # f = self.post_quantizer(f)
        f, post_state = self.post_rnn(f, post_state)
        return f, f_lens, pre_state, post_state


class Prediction(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed = torch.nn.Embedding(
            RNNTParam.num_labels-1,
            RNNTParam.pred_hidden_size
        )
        self.pred_rnn = torch.nn.LSTM(
            RNNTParam.pred_hidden_size,
            RNNTParam.pred_hidden_size,
            RNNTParam.pred_num_layers
        )
        self.sos = RNNTParam.SOS

    def forward(self, x: Tensor,
            state: Tuple[Tensor, Tensor],
            batch_size: int=1) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
          x: {U, N}
          state: ({D*L, N, C}, {D*L, N, C})

        Returns:
          g: {U, N, C}
          state: ({D*L, N, C}, {D*L, N, C})
        """
        # hack SOS, since there is no SOS in embedding table :(
        # TODO: fuse embedding + maskfill
        sos_mask = x.eq(self.sos)
        g = x.masked_fill(sos_mask, 0)
        g = self.embed(g)
        g = g.masked_fill_(sos_mask.unsqueeze(2), 0.0)
        g, state = self.pred_rnn(g, state)
        return g, state


class Joint(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear1_trans = Linear(
            RNNTParam.trans_hidden_size,
            RNNTParam.joint_hidden_size
        )
        self.linear1_pred = torch.nn.Linear(
            RNNTParam.pred_hidden_size,
            RNNTParam.joint_hidden_size
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            RNNTParam.joint_hidden_size,
            RNNTParam.num_labels
        )

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        """
        Args:
          f: {T=1, N, TC}
          g: {U=1, N, PC}

        Returns:
          y: {N, T=1, U=1, K}
        """
        y = self.linear1_trans(f)
        y += self.linear1_pred(g)
        y = self.relu(y)
        y = self.linear2(y)
        return y


class StackTime(torch.nn.Module):
    def __init__(self, stack_time_factor):
        super().__init__()
        self.stack_time_factor = stack_time_factor
        self.trans_hidden_size = RNNTParam.trans_hidden_size

    # TODO: opt reorder f: [T, N, TRANS_OC] => [T/2, N, TRANS_OC*2]
    def forward(self, x: Tensor, x_lens: Tensor) -> List[Tensor]:
        """
        Args:
          x: {T, N, C}
          x_lens: {N}

        Returns:
          x: {T/2, N, C*2}
          x_lens: {N}
        """
        r = torch.transpose(x, 0, 1)
        s = r.shape
        zeros = torch.zeros(
            s[0], (-s[1]) % self.stack_time_factor, s[2], dtype=r.dtype, device=r.device)
        r = torch.cat([r, zeros], 1)
        s = r.shape
        rs = [s[0], s[1] // self.stack_time_factor, s[2] * self.stack_time_factor]
        r = torch.reshape(r, rs)
        y = torch.transpose(r, 0, 1).contiguous()
        y_lens = torch.ceil(x_lens / self.stack_time_factor).long()

        for batch_idx in range(y.size(1)):
            if x_lens[batch_idx] % 2 == 1:
                y[y_lens[batch_idx]-1:, batch_idx, self.trans_hidden_size:] = 0
        return y, y_lens


import torch
from torch.nn import Linear
from rnn import LSTM
from typing import List, Tuple
from utils import *
from quant_modules import TensorQuantizer


class RNNTParam:
    # Transcription
    trans_input_size = 240  # 80*3
    trans_hidden_size = 1024
    pre_num_layers = 2
    post_num_layers = 3
    stack_time_factor = 2
    # Prediction
    pred_hidden_size = 320
    pred_num_layers = 2
    # Joint
    joint_hidden_size = 512
    num_labels = 29
    # [SOS, SPACE, a~z, ', BLANK]
    # [-1, 0, 1~26, 27, 28]
    SOS = -1
    BLANK = 28
    max_symbols = 30
    sample_rate = 16000


class GreedyDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.rnnt = model

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> List[List[int]]:
        f, f_lens = self.rnnt.transcription(x, x_lens)
        batch_size = f_lens.size(0)
        res = [[] for i in range(batch_size)]
        eos_idxs = (f_lens - 1).to(torch.int64)
        time_idxs = symbols_added = torch.zeros(batch_size, dtype=torch.int64)
        reach_max_idxs = torch.tensor([False]*batch_size)
        fi = f[0]
        pre_g = torch.zeros((1, batch_size), dtype=torch.int64)
        pre_hg = torch.zeros((RNNTParam.pred_num_layers, batch_size, RNNTParam.pred_hidden_size))
        pre_cg = torch.zeros((RNNTParam.pred_num_layers, batch_size, RNNTParam.pred_hidden_size))

        while True:
            g, (hg, cg) = self.rnnt.prediction(pre_g, (pre_hg, pre_cg))
            y = self.rnnt.joint(fi, g[0])
            symbols = torch.argmax(y, dim=1)
            no_blank_idxs = symbols.ne(RNNTParam.BLANK)
            # update res & g
            if torch.count_nonzero(no_blank_idxs) != 0:
                for i in range(batch_size):
                    if no_blank_idxs[i] == True:
                        res[i].append(symbols[i].item())
                symbols_added += no_blank_idxs
                reach_max_idxs = symbols_added.eq(RNNTParam.max_symbols)

                pre_g[0][no_blank_idxs] = symbols[no_blank_idxs]
                pre_hg[:, no_blank_idxs, :] = hg[:, no_blank_idxs, :]
                pre_cg[:, no_blank_idxs, :] = cg[:, no_blank_idxs, :]
            # update f
            if torch.count_nonzero(no_blank_idxs) != batch_size or torch.count_nonzero(reach_max_idxs) != 0:
                time_idxs += (~no_blank_idxs | reach_max_idxs)
                time_idxs = time_idxs.min(eos_idxs)  # TODO: add early response
                if torch.equal(time_idxs, eos_idxs):
                    break
                fetch_idxs = time_idxs.unsqueeze(1).unsqueeze(0).expand(
                    1, batch_size, RNNTParam.trans_hidden_size)
                fi = f.gather(0, fetch_idxs).squeeze(0)
                symbols_added *= no_blank_idxs
        return res


class RNNT(torch.nn.Module):
    def __init__(self, model_path=None, run_mode=None, split_fc1=False):
        super().__init__()
        self.transcription = Transcription()
        self.prediction = Prediction()
        self.joint = Joint(split_fc1)
        if model_path is not None:
            saved_quantizers = False if run_mode == None or run_mode == "calib" else True
            self._load_model(model_path, run_mode, saved_quantizers, split_fc1)

    def _load_model(self, model_path, run_mode=None, saved_quantizers=False, split_fc1=False):
        model = torch.load(model_path, map_location="cpu")
        state_dict = migrate_state_dict(model, split_fc1)
        if saved_quantizers:
            self.transcription.pre_rnn._init_quantizers(run_mode)
            self.transcription.post_rnn._init_quantizers(run_mode)
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)
            self.transcription.pre_rnn._init_quantizers(run_mode)
            self.transcription.post_rnn._init_quantizers(run_mode)
        self.transcription.pre_rnn._init_weights()
        self.transcription.post_rnn._init_weights()


class Transcription(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pre_rnn = LSTM(
            RNNTParam.trans_input_size,
            RNNTParam.trans_hidden_size,
            RNNTParam.pre_num_layers,
        )
        self.stack_time = StackTime(RNNTParam.stack_time_factor)
        self.post_rnn = LSTM(
            RNNTParam.trans_hidden_size*RNNTParam.stack_time_factor,
            RNNTParam.trans_hidden_size,
            RNNTParam.post_num_layers,
        )

    def forward(self, x: torch.Tensor,
            x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_quantizer: TensorQuantizer = self.pre_rnn._input_quantizers[0]
        x = pre_quantizer._quant_forward(x)
        y1, _ = self.pre_rnn(x, None)
        # TODO: eliminate contiguous after stack_time
        y2, f_lens = self.stack_time(y1, x_lens)
        post_quantizer: TensorQuantizer = self.post_rnn._input_quantizers[0]
        y2 = post_quantizer._quant_forward(y2.contiguous())
        f, _ = self.post_rnn(y2, None)
        return f, f_lens


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

    def forward(self, x: torch.Tensor,
            state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        y = self.embed(x)
        g, hidden = self.pred_rnn(y, state)
        return g, hidden


class Joint(torch.nn.Module):
    def __init__(self, split_fc1=False, **kwargs):
        super().__init__()
        self.split_fc1 = split_fc1
        if self.split_fc1:
            self.linear1_trans = Linear(
                RNNTParam.trans_hidden_size,
                RNNTParam.joint_hidden_size
            )
            self.linear1_pred = Linear(
                RNNTParam.pred_hidden_size,
                RNNTParam.joint_hidden_size
            )
        else:
            self.linear1 = Linear(
                RNNTParam.trans_hidden_size+RNNTParam.pred_hidden_size,
                RNNTParam.joint_hidden_size
            )
        self.relu = torch.nn.ReLU()
        self.linear2 = Linear(
            RNNTParam.joint_hidden_size,
            RNNTParam.num_labels
        )

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.split_fc1:
            y1 = self.linear1_trans(f)
            y1 += self.linear1_pred(g)
        else:
            x = torch.cat([f, g], dim=1)
            y1 = self.linear1(x)
        y2 = self.relu(y1)
        y = self.linear2(y2)
        return y


class StackTime(torch.nn.Module):
    def __init__(self, stack_time_factor):
        super().__init__()
        self.stack_time_factor = stack_time_factor

    # TODO: opt reorder f: [T, N, TRANS_OC] => [T/2, N, TRANS_OC*2]
    def forward(self, x: torch.Tensor,
            x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = torch.transpose(x, 0, 1)
        s = r.shape
        zeros = torch.zeros(
            s[0], (-s[1]) % self.stack_time_factor, s[2], dtype=r.dtype, device=r.device)
        r = torch.cat([r, zeros], 1)
        s = r.shape
        rs = [s[0], s[1] // self.stack_time_factor, s[2] * self.stack_time_factor]
        r = torch.reshape(r, rs)
        y = torch.transpose(r, 0, 1)
        y_lens = torch.ceil(x_lens.float() / self.stack_time_factor).int()
        return y, y_lens


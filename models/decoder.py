import os
import torch
from config import RNNTParam
from torch import Tensor
from typing import List
from utils import *


class GreedyDecoder(torch.nn.Module):
    def __init__(self, model, split_len=-1):
        super().__init__()
        self.rnnt = model
        self.split_len = split_len

    def forward(self, x: Tensor, x_lens: Tensor) -> List[List[int]]:
        """
        Args:
          x: {T, N, C}
          x_lens: {N}

        Returns:
          res: {N}
        """
        self.batch_size = x_lens.size(0)
        self.res = [[] for i in range(self.batch_size)]
        self.step = torch.zeros((self.batch_size, 2))  # debug only
        # init transcription tensors
        self.pre_state = None
        self.post_state = None
        # init prediction tensors
        self.pred_g = torch.tensor([[RNNTParam.SOS]*self.batch_size], dtype=torch.int64)
        self.pred_state = (
            torch.zeros((RNNTParam.pred_num_layers, self.batch_size, RNNTParam.pred_hidden_size)),
            torch.zeros((RNNTParam.pred_num_layers, self.batch_size, RNNTParam.pred_hidden_size)))

        if self.split_len != -1:
            max_len = x_lens.max().item()
            split_lens = torch.tensor(
                [self.split_len]*self.batch_size, dtype=torch.int64)
            for split_idx in range(0, max_len, self.split_len):
                # 0. split x, x_lens
                xi_lens = torch.min(
                    split_lens, torch.clamp((x_lens - split_idx), min=0))
                xi = x[split_idx : split_idx+self.split_len]
                self.greedy_decode(xi, xi_lens)
        else:
            self.greedy_decode(x, x_lens)
        return self.res

    def greedy_decode(self, f: Tensor, f_lens: Tensor) -> List[List[int]]:
        # init flags
        self.symbols_added = torch.zeros(self.batch_size, dtype=torch.int64)
        self.time_idx = torch.zeros(self.batch_size, dtype=torch.int64)
        self.finish = f_lens.eq(0)
        self.trace = [[0]*f_len for f_len in f_lens]  # debug only
        # 1. do transcription
        f, f_lens, self.pre_state, self.post_state = self.rnnt.transcription(f, f_lens, self.pre_state, self.post_state)
        fi = f[0]

        while True:
            # TODO: bf16 prediction + joint
            # 2. do prediction
            g, state = self.rnnt.prediction(self.pred_g, self.pred_state, self.batch_size)
            # TODO: fuse joint + argmax
            # 3. do joint
            y = self.rnnt.joint(fi, g[0])
            symbols = torch.argmax(y, dim=1)
            # 4. if (no BLANK and no MAX_SYMBOLS_PER_STEP) and no FINISH
            self.update_g = symbols.ne(RNNTParam.BLANK) & self.symbols_added.ne(RNNTParam.max_symbols_per_step) & ~self.finish
            if torch.count_nonzero(self.update_g) != 0:
                self.step[self.update_g, 1] += 1
                # 4.1. update res
                for batch_idx in range(self.batch_size):
                    if self.update_g[batch_idx]:
                        self.trace[batch_idx][self.time_idx[batch_idx]] += 1
                        self.res[batch_idx].append(symbols[batch_idx].item())
                # 4.2. update symbols_added
                self.symbols_added += self.update_g
                # 4.3. update g
                self.pred_g[0][self.update_g] = symbols[self.update_g]
                self.pred_state[0][:, self.update_g, :] = state[0][:, self.update_g, :]
                self.pred_state[1][:, self.update_g, :] = state[1][:, self.update_g, :]

            # 5. if (BLANK or MAX_SYMBOLS_PER_STEP) and no FINISH
            self.update_f = ~self.update_g & ~self.finish
            if torch.count_nonzero(self.update_f) != 0:
                self.step[self.update_f, 0] += 1
                # 5.1. update time_idx
                self.time_idx += self.update_f
                # 5.2. BCE
                self.finish |= self.time_idx.ge(f_lens)  # TODO: add early response
                self.time_idx = torch.clamp(self.time_idx.min(f_lens-1), min=0)  # BCE
                if torch.count_nonzero(self.finish) == self.batch_size:
                    break
                # 5.3. update f
                fetch_idx = self.time_idx.unsqueeze(1).unsqueeze(0).expand(
                    1, self.batch_size, RNNTParam.trans_hidden_size)
                fi = f.gather(0, fetch_idx).squeeze(0)
                # 5.4. reset symbols_added
                self.symbols_added *= ~self.update_f
        return self.res

    def _dump_tensors(self, **kwargs):
        torch.set_printoptions(precision=10)
        s = f"step: {self.step}\n"
        s += f"trace: {self.trace}\n"
        avg_freq = [sum(self.trace[batch_idx]) / len(self.trace[batch_idx]) if len(self.trace[batch_idx]) != 0 else 0 for batch_idx in range(self.batch_size)]
        s += f"avg update_g frequence: {avg_freq}\n"
        s += f"max update_g steps: {max(max(self.trace))}\n"
        s += f"symbols_added: {self.symbols_added}\n"
        s += f"time_idx: {self.time_idx}\n"
        if kwargs.get('f_lens') != None:
            s+= f"f_lens: {kwargs.get('f_lens')}\n"
        s += f"finish: {self.finish}\n"
        s += f"update_g: {self.update_g}\n"
        s += f"update_f: {self.update_f}\n"
        s += f"pred_g: {self.pred_g}\n"
        s += f"pred_hg.sum: {self.pred_state[0].contiguous().sum()}\n"
        s += f"pred_cg.sum: {self.pred_state[1].contiguous().sum()}\n"
        if kwargs.get('fi') != None:
            s += f"fi.sum: {kwargs.get('fi').contiguous().sum()}\n"
        s += f"pre_hx.sum: {self.pre_state[0].contiguous().sum()}\n"
        s += f"pre_cx.sum: {self.pre_state[1].contiguous().sum()}\n"
        s += f"post_hx.sum: {self.post_state[0].contiguous().sum()}\n"
        s += f"post_cx.sum: {self.post_state[1].contiguous().sum()}\n"
        if kwargs.get('y') != None:
            s += f"y.sum: {kwargs.get('y').contiguous().sum()}\n"
        s += "".join([f"{batch_idx}: {seq_to_sen(self.res[batch_idx])}\n" for batch_idx in range(self.batch_size)])
        print(s)
        print('-'*30)


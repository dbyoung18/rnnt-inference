import os
import torch
from config import RNNTParam
from torch import Tensor
from typing import List
from utils import *
import _C as P
import numpy as np


class GreedyDecoder(torch.nn.Module):
    def __init__(self, model, enable_bf16=False, split_len=-1):
        super().__init__()
        self.rnnt = model
        self.split_len = split_len
        self.enable_bf16 = enable_bf16

    def forward(self, x: Tensor, x_lens: Tensor):
        """
        Args:
          x: {T, N, C}
          x_lens: {N}

        Returns:
          res: {N}
        """
        self.batch_size = x_lens.size(0)
        self.res = torch.full((self.batch_size, RNNTParam.max_symbols_per_step*x_lens.max().item()), -1, dtype=torch.int32)
        self.res_idx = torch.full((self.batch_size,), -1, dtype=torch.int32)
        self.step = torch.zeros((self.batch_size, 2))  # debug only
        # TODO
        # python load_jit=true need to init pre_state and post_state
        # hx_pre, cx_pre = [], []
        # for i in range(RNNTParam.pre_num_layers):
        #     hx_pre.append(torch.zeros(x.size(1), RNNTParam.trans_hidden_size,
        #             dtype=torch.int8))
        #     cx_pre.append(torch.zeros(x.size(1), RNNTParam.trans_hidden_size,
        #             dtype=torch.float16))
        # self.pre_state = (hx_pre, cx_pre)
        # hx_post, cx_post = [], []
        # for i in range(RNNTParam.post_num_layers):
        #     hx_post.append(torch.zeros(x.size(1), RNNTParam.trans_hidden_size,
        #             dtype=torch.int8))
        #     cx_post.append(torch.zeros(x.size(1), RNNTParam.trans_hidden_size,
        #             dtype=torch.float16))
        # self.post_state = (hx_post, cx_post)
        # init transcription tensors
        self.pre_state = None
        self.post_state = None
        # init prediction tensors
        self.pred_g = torch.tensor([[RNNTParam.SOS]*self.batch_size], dtype=torch.int32)
        self.pred_hg = torch.zeros((RNNTParam.pred_num_layers, self.batch_size, RNNTParam.pred_hidden_size))
        self.pred_cg = torch.zeros((RNNTParam.pred_num_layers, self.batch_size, RNNTParam.pred_hidden_size))
        if self.enable_bf16:
            self.pred_hg, self.pred_cg = self.pred_hg.to(torch.bfloat16), self.pred_cg.to(torch.bfloat16)
        self.pred_state = [self.pred_hg, self.pred_cg]

        if self.split_len != -1:
            max_len = x_lens.max().item()
            split_lens = torch.tensor(
                [self.split_len]*self.batch_size, dtype=torch.int32)
            for split_idx in range(0, max_len, self.split_len):
                # 0. split x, x_lens
                xi_lens = torch.min(
                    split_lens, torch.clamp((x_lens - split_idx), min=0))
                xi = x[split_idx : split_idx+self.split_len]
                self.greedy_decode(xi, xi_lens)
        else:
            self.greedy_decode(x, x_lens)
        return self.res, self.res_idx+1

    def greedy_decode(self, f: Tensor, f_lens: Tensor):
        # init flags
        self.symbols_added = torch.zeros(self.batch_size, dtype=torch.int32)
        self.time_idx = torch.zeros(self.batch_size, dtype=torch.int32)
        # 1. do transcription
        f, f_lens, self.pre_state, self.post_state = self.rnnt.transcription(f, f_lens, self.pre_state, self.post_state)
        self.finish = f_lens.eq(0)
        f_lens = f_lens.to(torch.int32)
        if self.enable_bf16:
            f = f.to(torch.bfloat16)
        fi = f[0]

        while True:
            # TODO: bf16 prediction + joint
            # 2. do prediction
            g, state = self.rnnt.prediction(self.pred_g, self.pred_state)
            # 3. do joint
            y = self.rnnt.joint(fi, g[0])
            symbols = torch.argmax(y, dim=1)
            finish = P.greedy_decode_update(symbols, self.symbols_added, self.res, self.res_idx, self.time_idx, f_lens, self.pred_g, f, fi, self.pred_state[0], self.pred_state[1], state[0], state[1])

            if finish:
                break
        return self.res

    def _dump_tensors(self, **kwargs):
        torch.set_printoptions(precision=10)
        s = f"step: {self.step}\n"
        # s += f"trace: {self.trace}\n"
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
        s += "".join([f"{batch_idx}: {seq_to_sen(self.res[batch_idx], self.res_idx[batch_idx])}\n" for batch_idx in range(self.batch_size)])
        print(s)
        print('-'*30)


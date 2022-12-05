# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import math
import librosa
from .segment import AudioSegment
from utils import *


class WaveformFeaturizer(object):
    def __init__(self, input_cfg):
        self.cfg = input_cfg

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(file_path,
                                       target_sr=self.cfg['sample_rate'],
                                       int_values=self.cfg.get(
                                           'int_values', False),
                                       offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        return cls(input_config)


CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                             device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                            device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :seq_len[i].item()].mean()
            x_std[i] = x[i, :, :seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    return torch.cat(seq, dim=1)[:, :, ::frame_splicing]


class FilterbankFeatures(nn.Module):
    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hamming", normalize="per_feature", n_fft=None,
                 preemph=0.97,
                 nfilt=64, lowfreq=0, highfreq=None, log=True, dither=CONSTANT,
                 pad_to=8,
                 max_duration=16.7,
                 frame_splicing=1,
                 pad_output=False):
        super(FilterbankFeatures, self).__init__()

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size)  # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        # For now, always enable this.
        # See https://docs.google.com/presentation/d/1IVC3J-pHB-ipJpKsJox_SqmDHYdkIaoCXTbKmJmV2-I/edit?usp=sharing for elaboration
        self.use_deterministic_dithering = True
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False).float() if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq,
                                fmax=highfreq), dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad
        OUT_FEAT = 256 if pad_output else 240
        OUT_LEN = 500
        self.output_shape = torch.zeros((128, OUT_FEAT, OUT_LEN))

    def get_seq_len(self, seq_len):
        seq_len = torch.ceil(seq_len / self.hop_length).to(dtype=torch.int32)
        if self.frame_splicing > 1:
            seq_len = torch.ceil(seq_len / self.frame_splicing).to(dtype=torch.int32)
        return seq_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lens = self.get_seq_len(x_lens)

        # dither
        if self.dither > 0 and not self.use_deterministic_dithering:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        x = torch.ops.intel_mlperf.preemphasis(x, coeff=self.preemph)

        # do stft
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length,
            center=False, window=self.window,
		    return_complex=True)

        # get power spectrum
        x = x.abs().square()

        if self.dither > 0 and self.use_deterministic_dithering:
            x = x + self.dither ** 2
        # dot with filterbank energies
        x = torch.matmul(self.fb, x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        x = torch.ops.intel_mlperf.frame_splicing(x, self.frame_splicing)

        # normalize if required
        x = torch.ops.intel_mlperf.i_layernorm_pad(
            x, torch.ones_like(x), torch.zeros_like(x), x_lens, 1e-12, unbiased=1, output_shape=self.output_shape)

        return x, x_lens

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'], window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'], n_fft=cfg['n_fft'],
                   nfilt=cfg['features'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   max_duration=cfg.get('max_duration', 16.7),
                   dither=cfg['dither'], pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get('frame_splicing', 1), log=log, pad_output=cfg.get('pad_output'))


class FeatureFactory(object):
    featurizers = {
        "logfbank": FilterbankFeatures,
        "fbank": FilterbankFeatures,
    }

    def __init__(self):
        pass

    @classmethod
    def from_config(cls, cfg):
        feat_type = cfg.get('feat_type', "logspect")
        featurizer = cls.featurizers[feat_type]
        # return featurizer.from_config(cfg, log="log" in cfg['feat_type'])
        return featurizer.from_config(cfg, log="log" in feat_type)

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import array
import mlperf_loadgen as lg
import toml
import torch

from datasets.preprocessing import AudioPreprocessing
from rnnt_qsl import RNNTQSL
from utils import *


class PytorchSUT:
    def __init__(self, model_path, dataset_dir, batch_size=1, args=None, **kwargs):
        # create preprocessor
        if args.enable_preprocess and os.path.exists(args.toml_path):
            config = toml.load(args.toml_path)
            featurizer_config = config["input_eval"]
            self.preprocessor = AudioPreprocessing(**featurizer_config).eval()
        else:
            self.preprocessor = None
        # create model
        if args.run_mode == "quant":
            from modeling_rnnt_quant import RNNT, GreedyDecoder
        else:
            from modeling_rnnt import RNNT, GreedyDecoder
        rnnt = RNNT(model_path, args.run_mode, args.split_fc1).eval()
        self.model = GreedyDecoder(rnnt)
        self.enable_preprocess = (self.preprocessor != None)
        self.use_jit = args.jit
        self.batch_size = batch_size
        self.scenario = args.scenario
        if self.use_jit:
            if self.enable_preprocess:
                self.preprocessor = jit_module(self.preprocessor)
            self.model.rnnt = jit_model(self.model.rnnt)
        # create qsl & sut
        self.qsl = RNNTQSL(dataset_dir)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, samples):
        if self.scenario == "Offline":
            samples.sort(key=lambda s: self.qsl[s.index][1].item(), reverse=True)
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i : min(i+self.batch_size, len(samples))]
            batch_idx = [sample.index for sample in batch_samples]
            results = self.inference(batch_idx)
            self.query_samples_complete(batch_samples, results)

    def inference(self, batch_idx):
        with torch.no_grad():
            if self.enable_preprocess:
                wavs = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx], batch_first=True)
                wav_lens = torch.tensor(
                    [self.qsl[idx][1] for idx in batch_idx])
                feas, fea_lens = self.preprocessor(wavs, wav_lens)
            else:
                feas = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx])
                fea_lens = torch.tensor(
                    [self.qsl[idx][1] for idx in batch_idx])
            results = self.model(feas, fea_lens)
        return results

    def query_samples_complete(self, samples, results):
        batch_responses = []
        for i in range(len(results)):
            res_arr = array.array("q", results[i])
            buf_inf = res_arr.buffer_info()
            response = lg.QuerySampleResponse(
                samples[i].id,
                buf_inf[0],
                buf_inf[1]*res_arr.itemsize
            )
            lg.QuerySamplesComplete([response])
            # batch_responses.append(response)
            logger.debug(f"{samples[i].index}::{seq_to_sen(results[i])}")
        # lg.QuerySamplesComplete(batch_responses)
    
    def flush_queries(self):
        pass

    def __del__(self):
        lg.DestroySUT(self.sut)
        print("Finished destroying SUT.")


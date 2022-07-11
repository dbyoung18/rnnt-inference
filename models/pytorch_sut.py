import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import array
import mlperf_loadgen as lg
import torch

from datasets.preprocessing import AudioPreprocessing
from rnnt import *
from rnnt_qsl import RNNTQSL
from utils import *


class PytorchSUT:
    def __init__(self, model_path, manifest_path, dataset_dir, **kwargs):
        use_jit = kwargs.pop("use_jit", False)
        run_mode = kwargs.pop("run_mode", None)
        perf_count = kwargs.pop("perf_count", None)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.qsl = RNNTQSL(dataset_dir, perf_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        rnnt = RNNT(model_path, run_mode).eval()
        #  if run_mode == "fake_quant" or run_mode == "quant":
            #  rnnt._init_scales(kwargs.pop("calib_path", None))
        if use_jit:
            rnnt = jit_model(rnnt)
        self.model = GreedyDecoder(rnnt)

    def issue_queries(self, samples):
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i : min(i+self.batch_size, len(samples))]
            batch_idx = [sample.index for sample in batch_samples]
            results = self.inference(batch_idx)
            self.query_samples_complete(batch_samples, results)

    def inference(self, batch_idx):
        feas = torch.nn.utils.rnn.pad_sequence(
            [self.qsl[idx][0] for idx in batch_idx])
        fea_lens = torch.tensor(
            [self.qsl[idx][1] for idx in batch_idx])
        with torch.no_grad():
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


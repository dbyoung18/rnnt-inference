import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import array
import mlperf_loadgen as lg
import torch

from utils import *


class PytorchSUT:
    def __init__(self, model, qsl, batch_size=1, **kwargs):
        self.batch_size = batch_size
        self.model = model
        self.qsl = qsl
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

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


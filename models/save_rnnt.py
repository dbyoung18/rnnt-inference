import argparse
import mlperf_loadgen as lg
import os
import subprocess
import torch

from pytorch_sut import PytorchSUT
from modeling_rnnt_quant import *
from rnnt_qsl import RNNTQSL
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--jit", action="store_true", help="enable jit")
    parser.add_argument("--split_fc1", action="store_true", help="split joint linear1")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rnnt = RNNT(args.model_path, run_mode="quant", split_fc1=args.split_fc1).eval()
    model = GreedyDecoder(rnnt)
    qsl = RNNTQSL(args.dataset_dir)
    # create sut & qsl 
    sut = PytorchSUT(model, qsl, args.batch_size)
    print("==> Freezing model...")
    # quant & prepack
    results = sut.inference(range(0, args.batch_size))
    for i in range(len(results)):
        logger.debug(f"{i}::{seq_to_sen(results[i])}")
    quant_model_path = os.path.join(os.path.dirname(args.model_path), "rnnt_quant.pt")
    torch.save(sut.model.rnnt.state_dict(), quant_model_path) 
    # jit
    if args.jit:
        sut.model.rnnt = jit_model(sut.model.rnnt)
        jit_model_path = os.path.join(os.path.dirname(args.model_path), "rnnt_jit.pt")
        torch.save(sut.model.rnnt.state_dict(), jit_model_path)

if __name__ == "__main__":
    main()


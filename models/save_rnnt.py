import argparse
import mlperf_loadgen as lg
import os
import subprocess
import torch
import toml

from datasets.preprocessing import AudioPreprocessing
from modeling_rnnt_quant import *
from pytorch_sut import PytorchSUT
from rnnt_qsl import RNNTQSL
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--toml_path", type=str, default="../configs/rnnt.toml")
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--run_mode", default="quant",
        choices=[None, "calib", "quant", "fake_quant"], help="run_mode, default quant")
    parser.add_argument("--jit", action="store_true", help="enable jit")
    parser.add_argument("--split_fc1", action="store_true", help="split joint linear1")
    parser.add_argument("--enable_preprocess", action="store_true", help="enable audio preprocess")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # create sut & qsl
    sut = PytorchSUT(args.model_path, args.dataset_dir, args.batch_size, args)
    # jit preprocessor
    if args.jit and args.enable_preprocess:
        print("==> JIT audio preprocessor...")
        sut.preprocessor = jit_module(sut.preprocessor)
        jit_preprocessor_path = os.path.join(os.path.dirname(args.model_path), "preprocessor_jit.pt")
        torch.jit.save(sut.preprocessor, jit_preprocessor_path)
    # quant & prepack model
    print("==> Freezing model...")
    results = sut.inference(range(0, args.batch_size))
    for i in range(len(results)):
        logger.debug(f"{i}::{seq_to_sen(results[i])}")
    quant_model_path = os.path.join(os.path.dirname(args.model_path), "rnnt_quant.pt")
    torch.save(sut.model.rnnt.state_dict(), quant_model_path) 
    # jit model
    if args.jit:
        print("==> JIT model...")
        sut.model.rnnt = jit_model(sut.model.rnnt)
        jit_model_path = os.path.join(os.path.dirname(args.model_path), "rnnt_jit.pt")
        torch.jit.save(sut.model.rnnt, jit_model_path)

if __name__ == "__main__":
    main()


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
    parser.add_argument("--scenario", default="Offline",
        choices=["SingleStream", "Offline", "Server"], help="Scenario")
    parser.add_argument("--toml_path", type=str, default="../configs/rnnt.toml")
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--run_mode", default="quant",
        choices=["f32", "calib", "quant", "fake_quant"], help="run_mode, default quant")
    parser.add_argument("--load_jit", action="store_true", help="load jit")
    parser.add_argument("--save_jit", action="store_true", help="save jit")
    parser.add_argument("--enable_preprocess", action="store_true", help="enable audio preprocess")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # create sut & qsl
    sut = PytorchSUT(args.model_path, args.dataset_dir, args.batch_size, args)
    # save preprocessor & model
    suffix = f"_{args.run_mode}_jit" if args.save_jit else f"_{args.run_mode}"
    if args.save_jit:
        if args.enable_preprocess:
            print("==> JIT audio preprocessor...")
            preprocessor_path = os.path.join(os.path.dirname(args.model_path), "preprocessor_jit.pt")
            torch.jit.save(sut.preprocessor, preprocessor_path)

        print("==> JIT model...")
        model_path = os.path.join(os.path.dirname(args.model_path), f"rnnt{suffix}.pt")
        torch.jit.save(sut.model.rnnt, model_path)
    else:
        if args.enable_preprocess:
            print("==> Save audio preprocessor...")
            preprocessor_path = os.path.join(os.path.dirname(args.model_path), "preprocessor.pt")
            torch.save(sut.preprocessor, preprocessor_path)

        print("==> Save model...")
        model_path = os.path.join(os.path.dirname(args.model_path), f"rnnt{suffix}.pt")
        torch.save(sut.model.rnnt.state_dict(), model_path)

if __name__ == "__main__":
    main()


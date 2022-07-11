import argparse
import mlperf_loadgen as lg
import os
import subprocess
import torch

from pytorch_sut import PytorchSUT
#  from quant_rnnt import QuantRNNT

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf_count", type=int, default=None, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--calib_path", type=str, default=None)
    parser.add_argument("--run_mode", default=None,
        choices=[None, "calib", "quant", "fake_quant", "dynamic_quant"], help="run_mode, default fp32")
    parser.add_argument("--jit", action="store_true", help="enable jit")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # create sut & qsl 
    sut = PytorchSUT(
        args.model_path, args.manifest_path, args.dataset_dir,
        perf_count=args.perf_count,
        batch_size=args.batch_size,
        run_mode=args.run_mode
    )
    # calibration
    print("==> Running calibration...")
    for i in range(0, sut.qsl.count, args.batch_size):
        sut.inference(range(i, min(i+args.batch_size, sut.qsl.count)))
    model_path = "rnnt_test.pt"
    torch.save(sut.model.rnnt.state_dict(), model_path)
    #  calib_dict = save_calib(args.calib_path, sut.model)
    #  print(calib_dict)

    print("==> Freezing model...")
    # quant & prepack
    #  sut.model = QuantRNNT(model_path).eval()
    #  sut.inference(range(0, args.batch_size))
 
    # jit
    # sut.model.rnnt = jit_model(sut.model.rnnt)

if __name__ == "__main__":
    main()


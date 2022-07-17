import argparse
import mlperf_loadgen as lg
import os
import subprocess

from pytorch_sut import PytorchSUT
from rnnt_qsl import RNNTQSL
from eval_accuracy import eval_acc
from utils import *

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf_count", type=int, default=None, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--scenario", default="Offline",
        choices=["SingleStream", "Offline", "Server"], help="Scenario")
    parser.add_argument("--mlperf_conf", default="../configs/mlperf.conf",
        help="mlperf rules config")
    parser.add_argument("--user_conf", default="../configs/user.conf",
        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--toml_path", type=str, default="../configs/rnnt.toml")
    parser.add_argument("--model_path", type=str, default="work_dir/rnnt.pt")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--calib_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--run_mode", default=None,
        choices=[None, "calib", "quant", "fake_quant"], help="run_mode, default fp32")
    parser.add_argument("--jit", action="store_true", help="enable jit")
    parser.add_argument("--split_fc1", action="store_true", help="split joint linear1")
    parser.add_argument("--enable_preprocess", action="store_true", help="enable audio preprocess")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy evaluation")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.run_mode == "quant":
        from modeling_rnnt_quant import RNNT, GreedyDecoder
    else:
        from modeling_rnnt import RNNT, GreedyDecoder
    rnnt = RNNT(args.model_path, run_mode=args.run_mode, split_fc1=args.split_fc1).eval()
    model = GreedyDecoder(rnnt)
    qsl = RNNTQSL(args.dataset_dir)
    # create sut & qsl 
    sut = PytorchSUT(model, qsl, args.batch_size, args.enable_preprocess, args.toml_path)
    # set cfg
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
    settings.FromConfig(args.user_conf, "rnnt", args.scenario)
    settings.mode = lg.TestMode.AccuracyOnly if args.accuracy \
        else lg.TestMode.PerformanceOnly
    # set log
    os.makedirs(args.log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    print("==> Running loadgen test")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
    # eval accuracy
    if args.accuracy:
        print(f"==> Evaluating accuracy")
        log_path = os.path.join(args.log_dir, "mlperf_log_accuracy.json")
        acc_path = os.path.join(os.getcwd(), "eval_accuracy.py")
        cmd = f"python {acc_path} --log_path {log_path} --manifest_path {args.manifest_path}"
        print(f"==> Running accuracy script: {cmd}")
        subprocess.check_call(cmd, shell=True)
    print("Done!")

if __name__ == "__main__":
    main()


import torch
from models.rnnt import RNNT

if __name__ == "__main__":
    model_path = None
    calib_path = None
    run_mode = "fake_quant"
    rnnt = RNNT(model_path, calib_path, run_mode)
    rnnt.load_state_dict(torch.load("rnnt_test.pt"))
    rnnt.eval()
    print(rnnt)


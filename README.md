# RNNT MLPerf Inference BKC

## HW & SW requirements
###
```
  SPR 2 sockets
  GCC >= 11
```

## Steps to run RNNT

### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  export PATH=~/anaconda3/bin:$PATH
```

### 2. End-to-end run inference
Execute `run.sh`. The end-to-end process including:
| STAGE(default -2) | STEP |
|  -  | -  |
| -2 | Prepare conda environment |
| -1 | Prepare environment |
| 0 | Download model |
| 1 | Download dataset |
| 2 | Pre-process dataset |
| 3 | Calibration |
| 4 | Build model |
| 5 | Run Offline/Server accuracy & benchmark |

You can also use the following command to start with your custom conda-env/work-dir/step.
```
  ./run.sh [CONDA_ENV] [WORK_DIR] [STAGE]
```
* Currently, only support Offline scenario.

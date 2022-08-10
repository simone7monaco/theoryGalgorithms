import os
import sys
import argparse

from PGNN.models.train_pgnn import get_args as pgnn_args
from PGNN.models.train_pgnn import main as pgnn_main


exp_list = ["pgnn0", "pgnn", "pga0", "pga", "cbrain1", "cbrain2", "cbrain3", "dpgn0", "dpgn"]

experiment = sys.argv[1]
assert experiment in exp_list, f"The first argument must be the experiment name (one among {', '.join(exp_list)})"


if experiment == "pgnn":
    os.chdir("PGNN/models/")
    script = 'python train_pgnn.py '
elif experiment == "pgnn0":
    os.chdir("PGNN/models/")
    script = 'python train_pgnn.py --lamda=0'
elif experiment == "pga":
    os.chdir("PGA_LSTM/Models/")
    script = 'python Mendota_PGA_LSTM.py '
elif experiment == "cbrain1":
    os.chdir("CBRAIN_CAM")
    script = 'python cbrain_train.py -c nn_config/001_8col_pnas.yml '
elif experiment == "cbrain2":
    os.chdir("CBRAIN_CAM")
    script = 'python cbrain_train.py -c nn_config/002_8col_strong.yml '
elif experiment == "cbrain3":
    os.chdir("CBRAIN_CAM")
    script = 'python cbrain_train.py -c nn_config/003_8col_weak.yml '
elif experiment == "dpgn":
    os.chdir("dpgn")
    script = 'python main.py -f yaml/LA-DPGN.yaml '
elif experiment == "dpgn0":
    os.chdir("dpgn")
    script = 'python main.py -f yaml/LA-GNonly.yaml '


if len(sys.argv) > 2:
    script = script + " ".join(sys.argv[2:])

os.system(script)

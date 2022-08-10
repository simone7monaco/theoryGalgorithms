import os; os.chdir("/home/students/s265780/theory-G/Cophy-PGNN/notebooks")
import sys
sys.path.append('../scripts/')

import torch
from training import Trainer
from presets import LambdaSearch
from utils import *

from fastprogress.fastprogress import master_bar, progress_bar

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
param_presets = LambdaSearch(data_path='Cophy-PGNN/datasets/')

train_size = 200
break_loop = False
loss_plot = False

param = param_presets.NSE_DNNex_LB()
param.name = 'NSE-DNNex-LF'
param.data_params['train_size'] = train_size
param.train_params['break_loop_early'] = break_loop
param.loss_params['lambda_s'] = 0.566698
param.loss_params['lambda_e0'] = 3.680050
param.loss_params['anneal_factor'] = 0.786220
param.data_params['device'] = device
param.nn_params['device'] = device

print(os.getcwd())
trainer = Trainer(master_bar=None, plot=loss_plot)
trainer.start(param)

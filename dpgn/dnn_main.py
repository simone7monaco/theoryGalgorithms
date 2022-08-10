import logging
import os
from pathlib import Path
import pprint
import yaml
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from DNN import DNN as Net
from main import load_cfg
import wandb


TIME_DIM = 384
LAT_DIM = 141    # vertical
LONG_DIM = 129   # horizontal

def transform_3d_to_2d(X):
    return X.reshape(-1, X.shape[2])

def transform_2d_to_3d(X, num_nodes):
    return X.reshape((int(X.shape[0] / num_nodes), num_nodes, X.shape[1]));

def load_data(cfg):
    X = np.load(cfg['dataset']['X_path']) # Matrix in the form: (t, point, measurement)
    edge_index = np.load(cfg['dataset']['edge_index_path'])
    edge_attr = np.load(cfg['dataset']['edge_attr_path'])

#     edge_index = torch.tensor(edge_index)
#     edge_attr = torch.tensor(edge_attr)
    num_nodes = X.shape[1]
    
    Y = X[:, :,:1]
    for i in range(num_nodes): # adding number of edges
        X[:,i,:] = np.hstack([X[:,i,:-1], 
                              (edge_index[0] == i).sum()*np.ones([X.shape[0], 1])
                             ])
    Y = transform_3d_to_2d(Y)
    X = transform_3d_to_2d(X)
    return X, Y, num_nodes
    
    
def main(cfg, iteration, tr_frac):
    MODE = cfg['model']['MODE']    # 1: DNN, 2: Physics-only, 3: all
    MODE_DESC = cfg['model']['MODE_desc']
    REGION = cfg['dataset']['REGION']    # LA or SD
    PDE = cfg['model']['PDE']
    NN = cfg['model']['NN']
    USE_INPUT_PRED_LOSS = cfg['model']['USE_INPUT_PRED_LOSS']
    pred_input_weight = cfg['model']['pred_input_weight']
    SKIP = cfg['model']['SKIP']

    dirname = f"{MODE_DESC}_{REGION}_NN{NN}_tfrac_{tr_frac}_{iteration}"
    logdir = os.path.join("log", dirname)
    modeldir = os.path.join("model", dirname)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    if len(list(Path(logdir).iterdir())) > 5: return
    logfilename = os.path.join(logdir, 'log.txt')
    
    # Print the configuration - just to make sure that you loaded what you wanted to load
    with open(logfilename, 'w') as f:
        pp = pprint.PrettyPrinter(indent=4, stream=f)
        pp.pprint(cfg)
    
    logging.basicConfig(filename=logfilename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    writer = SummaryWriter(logdir)
    
    X, Y, num_nodes = load_data(cfg)
    
    ########## Device setting ##########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ####################################
    
    ########## Architecture setting ##########
    input_size = X.shape[1]
# #     edge_attr_size = 1    # embedding index
#     edge_num_embeddings = torch.max(edge_attr).item() + 2
#     edge_hidden_size = cfg['model']['edge_dim']
#     node_hidden_size = cfg['model']['node_dim']
#     global_hidden_size = cfg['model']['global_dim']
    output_size = 1    # predict Temperature
#     D = torch.tensor(cfg['model']['diff']).to(device)
#     sp_L = get_laplacian(edge_index, type="norm").to(device)
    ##########################################

    num_processing_steps = cfg['train']['num_processing_steps']    # Forecast horizon
    num_iterations = cfg['train']['num_iter']

    losses_sup = []    # supervised loss
    losses_phy = []    # physics loss
    losses_tot = []    # total loss
    val_losses_sup = []
    used_timestamps = []

    #### Model ####
    model = Net(
        input_size=input_size,
        hidden_size=cfg['model']['hidden_size'],
        output_size=output_size,
        depth=cfg['model']['depth'],
        drop_frac=0.1,
        act=torch.nn.ReLU,
        softmax=False)
    
    model.to(device)
    
    if wandb.run is not None:
        wandb.watch(model, log_freq=100)

    # Training loss
    criterion_mse = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg['optimizer']['initial_lr'], 
                           weight_decay=cfg['optimizer']['weight_decay'])
    reg_coeff = cfg['train']['reg_coeff']

    max_tr_ind = 250*num_nodes
    tr_ind, val_ind, te_ind = max_tr_ind*tr_frac, 300*num_nodes, (TIME_DIM-1)    # training/validation/test split
    tr_ind = int(tr_ind)
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    train_ds = TensorDataset(X[:tr_ind], Y[:tr_ind])
    val_ds = TensorDataset(X[max_tr_ind:val_ind], Y[max_tr_ind:val_ind])
    test_ds = TensorDataset(X[val_ind:-1], Y[val_ind:-1])
    
    print(f"Dataset portion {tr_frac} ({len(train_ds)}/{len(X)} samples in training)")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8)

    print(f"Train dataloader: {len(train_dl)}")
    print(f"Val dataloader: {len(val_dl)}")
    print(f"Test dataloader: {len(test_dl)}\n\n")

    
    #### Training
    for iter_ in trange(num_iterations, desc=dirname):
        model.train()
        for bX, bY in train_dl:
            bX, bY = bX.to(device, non_blocking=True), bY.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(bX)
            #### Training loss across processing steps.
            loss_sup = criterion_mse(output, bY)

            #### Physics rule
#             loss_phy_seq = [torch.sum((dt-ds)**2) for dt, ds in zip(time_derivatives, spatial_derivatives)]
#             loss_phy = sum(loss_phy_seq) / len(loss_phy_seq)
#             if USE_INPUT_PRED_LOSS:
#                 #### Use pred_inputs for optimization
#                 loss_pred_inputs_seq = [torch.sum((pred_input - torch.tensor(X[t+1+step_t,:,1:], dtype=torch.float32, device=device))**2)
#                                         for step_t, pred_input in enumerate(pred_inputs)]
#                 loss_pred_inputs = sum(loss_pred_inputs_seq) / len(loss_pred_inputs_seq)
#                 loss_sup = loss_sup + pred_input_weight*loss_pred_inputs

            #### loss
            if MODE == 1:
                loss = loss_sup
            elif MODE == 2:
                loss = loss_phy
            elif MODE == 3:
                pippo
                loss = loss_sup + reg_coeff*loss_phy

            
            #### Backward and optimize
            loss.backward()
            optimizer.step()

            
            losses_tot.append(loss.item())
            writer.add_scalars('loss/train', 
                               {'loss_tot': losses_tot[-1]}, iter_)

    #         losses_phy.append(loss_phy.item())
#             writer.add_scalars('loss/train', {'loss_sup': losses_sup[-1]}, iter_)
            
            if wandb.run is not None:
                wandb.log({'lr': optimizer.param_groups[0]['lr']})
                wandb.log({'loss/train': losses_tot[-1]})


        #### Validation
        if iter_%cfg['train']['valid_iter'] == 0:
            model.eval()
            losses_val = []
#             for vt in range(tr_ind, val_ind - num_processing_steps):
            with torch.no_grad():
                for bX, bY in val_dl:
                    bX, bY = bX.to(device, non_blocking=True), bY.to(device, non_blocking=True)
                    output = model(bX)
                    val_loss_sup = criterion_mse(output, bY)
                    losses_val.append(val_loss_sup.item())
                    
                if (len(val_losses_sup)>0) and (np.mean(losses_val)<np.min(val_losses_sup)):
                    # When best validation is found, check test set
                    losses_te = []
                    for bX, bY in val_dl:
                        bX, bY = bX.to(device, non_blocking=True), bY.to(device, non_blocking=True)
                        output = model(bX)
                        test_loss_sup = criterion_mse(output, bY)
                        losses_te.append(test_loss_sup.item())
                        
                    writer.add_scalars('loss/test', {'loss_sup': np.mean(losses_te)}, iter_)
                    if wandb.run is not None:
                        wandb.log({'loss/test': np.mean(losses_te)})
                        
                    logging.info("{}/{} iterations.".format(iter_, num_iterations))
                    logging.info("[Train]Loss: {:.4f}\t[Valid]Loss_sup: {:.4f}\t[Test]Loss_sup: {:.4f}"
                             .format(loss, np.mean(losses_val), np.mean(losses_te)))

                val_losses_sup.append(np.mean(losses_val))
                writer.add_scalars('loss/valid', {'loss_sup': val_losses_sup[-1]}, iter_)
                if wandb.run is not None:
                    wandb.log({'loss/valid': val_losses_sup[-1]})
    
                  
def new_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file (YAML format)",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu number: 0 or 1")
    
    parser.add_argument("--model_path",
                        dest="modelpath",
                        help="load pretrained model",
                        default=False)
    parser.add_argument("--use_wandb", 
                        type=bool, 
                        default=False, 
                        help="Log training to Wandb.")
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001, 
                        help="Log training to Wandb.")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=512, 
                       )
    parser.add_argument("--hidden_size", 
                        type=int, 
                        default=20, 
                       )
    parser.add_argument("--depth", 
                        type=int, 
                        default=5, 
                       )
    return parser

if __name__=="__main__":
    args = new_parser().parse_args()
    
    cfg = load_cfg(args.filename)
    batch_size = args.batch_size

#     torch.cuda.set_device(args.gpu)
    
    cfg['modelpath'] = args.modelpath
    cfg['optimizer']['initial_lr'] = args.lr
    cfg['model']['hidden_size'] = args.hidden_size
    cfg['model']['depth'] = args.depth
    
    tr_frac_range = [1, 0.5, 0.4, 0.3, 0.2, 0.1]

    
    if args.use_wandb:
        wandb.init(project="theoryG", entity="smonaco", 
                config={"hyper": "parameter"}, 
                tags=["dpgn_dnn"],
                settings=wandb.Settings(start_method='fork')
                )
    for iteration in range(1):
        for tr_frac in tr_frac_range:
            if args.use_wandb:
                wandb.init(project="theoryG", entity="smonaco", 
                        config={"hyper": "parameter"}, 
                        tags=["dpgn/dnn"],
                        settings=wandb.Settings(start_method='fork')
                        )
            main(cfg, iteration, tr_frac)
            if args.use_wandb:
                wandb.finish()

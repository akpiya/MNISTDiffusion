import numpy as np 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unet import *
from diffusion import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda'
default_dir = "../data/"


def get_data():
    ins = torch.Tensor(np.float32(np.load(default_dir+'MNIST_data.npy'))) / 255
    labels = (torch.Tensor(np.float32(np.load(default_dir+'MNIST_labels.npy')))
        .reshape(-1, 1))
    data = torch.hstack([ins, labels])
    train = data[:50000]
    val = data[50000:55000]
    test = data[55000:]
    return train, val, test

def convert_to_dataloader(train, val, test, config):
    a = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    b = DataLoader(val, batch_size=val.shape[0], shuffle=True)
    c = DataLoader(test, batch_size=test.shape[0],shuffle=True)
    return a,b,c


def create_config_highvar():
    """Configuration dictionary for the non-optimal diffusion model"""
    config = {}
    config['T'] = 200
    config['batch_size'] = 100
    config['beta_min'] = 0.0001
    config['beta_max'] = 0.1
    config['device'] = 'cuda'
    config['num_epochs'] = 60
    config['model'] = 'standard' # either 'standard' or 'epsilon'
    return config


def run_epochs_high_var(config):
    """Trains the non-optimal diffusion model with the configuration given by `create_config_highvar`"""
    train, val, test = convert_to_dataloader(*get_data(),config)
    snet = ScoreNet().to(config['device'])
    diff = Diffusion(snet,
                     config['T'],
                     config['device'],
                     config['beta_min'],
                     config['beta_max']).to(config['device'])
    
    #optimizer = torch.optim.Adam(diff.parameters())
    rho_params = []
    other_params = []

    for name, param in diff.named_parameters():
        if name == "mu.rho":
            rho_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {
            'params' : rho_params,
            'lr' : 2,
        },
        {
            'params' : other_params,
            'lr' : 0.001,
        }
    ])

    
    for epoch in range(config['num_epochs']):
        diff.train()
        train_loss = []
        # changing the learning rate for rho
        if epoch % 20 == 0:
            optimizer.param_groups[0]['lr'] /= 10
            print(f"current lr: {optimizer.param_groups[0]['lr']}")
        
        for batch_idx, batch in enumerate(train):
            x = batch[:, :-1].to(config['device'])
            x = x.reshape(-1, 1, 28, 28)
            y = batch[:, -1]
            xt_minus, xt, mu, ts = diff.forward_high_var(x)
            optimizer.zero_grad()
            loss = torch.mean(diff.mse_loss_high_var(xt_minus, mu, ts))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
            
        with torch.no_grad():
            diff.eval()
            data = next(iter(val))
            x = data[:, :-1]
            x = x.reshape(-1, 1, 28, 28).to(config['device'])
            y = data[:, -1]
            xt_minus, xt, mu, ts = diff.forward_high_var(x)

            loss = torch.mean(diff.mse_loss_high_var(xt_minus, mu, ts))
            print(f"Epoch {epoch} | Train Loss: {torch.mean(torch.Tensor(train_loss))} | Val Loss: {loss.item()}")
    return diff
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
import numpy as np
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
import VAE_dna_meth_model

# import csv
dna_meth = pd.read_csv('datasets_transpose_csv/dna_meth_transpose.csv')
dna_meth = dna_meth.drop(columns=['Unnamed: 0'])
cell_lines = dna_meth['CpG_sites_hg19']
dna_meth = dna_meth.drop(columns=['CpG_sites_hg19'])
dna_meth = dna_meth.drop(columns=['Unnamed: 81039'])
dna_meth_np = dna_meth.to_numpy()

# convert to tensor
dna_meth_full = torch.Tensor(dna_meth_np)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define dataloaders
batch_size = 32
full_loader = DataLoader(dna_meth_full,batch_size=batch_size)

# initialise model parameters
input_size = 81037 #dimension of gene expressions
level_2 = 2048
level_3 = 1500
latent_dim = 128 # target latent size
model = VAE_dna_meth_model.VAE(input_size, level_2, level_3, latent_dim).to(device)

# load model
model.load_state_dict(torch.load('trained_models/VAE_dna_meth' + str(latent_dim) + '.pt'))
model.eval()

# execute reducing of dimensions using VAE
with torch.no_grad():
    z_store = torch.zeros(1, latent_dim).to(device)
    for batch_index, data in enumerate(full_loader):
        data = data.to(device)
        x_hat, mu, logvar, z = model(data)
        z_store = torch.cat((z_store, z), 0)

all_data_z = z_store[1:]
all_data_z_np = all_data_z.cpu().numpy()

input_path = "C:/Users/mdzak/Desktop/GitHub/FYP_Zaki/results"
input_path_name = input_path.split('/')[-1]
latent_space_path = 'results/' + input_path_name + str(latent_dim) + 'D_latent_space_dna_meth.tsv'

all_data_z_df = pd.DataFrame(all_data_z_np)
all_data_z_df.to_csv(latent_space_path, sep='\t')
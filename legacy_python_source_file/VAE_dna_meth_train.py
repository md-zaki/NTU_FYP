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

# split train and test
dna_meth_np = dna_meth.to_numpy()
dna_meth_train, dna_meth_test = train_test_split(dna_meth_np, test_size=0.10, random_state=33, shuffle=False)

# convert to tensor
dna_meth_train_t = torch.Tensor(dna_meth_train)
dna_meth_test_t = torch.Tensor(dna_meth_test)
dna_meth_full = torch.Tensor(dna_meth_np)

# declare training parameters
epochs = 100
batch_size = 32
lr = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define dataloaders
train_loader = DataLoader(dna_meth_train_t, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dna_meth_test_t,batch_size=batch_size, shuffle=False)
full_loader = DataLoader(dna_meth_full,batch_size=batch_size, shuffle=False)

# initialise model parameters
input_size = 81037 #dimension of gene expressions
level_2 = 2048
level_3 = 1500
latent_dim = 1024 # target latent size
torch.manual_seed(33)
model = VAE_dna_meth_model.VAE(input_size, level_2, level_3, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# define loss function
def loss_function(x_hat, x, mean, log_var): # recon loss and kld loss
        bce = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction = 'sum')
        kld = 0.5 * torch.sum(log_var.exp() + mean.pow(2) - 1 - log_var)
        loss = kld + bce
        return loss

# define train function
def train (model, dataloader, epoch):
    model.train()
    train_loss = 0.0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        x_hat, mu, logvar, z = model(data)
        loss = loss_function(x_hat, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i%24 == 0:
            print("Train Epoch {} [Batch {}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), loss.item()/len(data)))

    print('=====> Epoch {}, Average Loss: {:.3f}'.format(epoch, train_loss/len(train_loader.dataset)))

# define validate function
def validate(model, dataloader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        mean_store = torch.zeros(1, latent_dim).to(device)
        for batch_index, data in enumerate(dataloader):
            data = data.to(device)
            x_hat, mu, logvar, z = model(data)
            test_loss += loss_function(x_hat, data, mu, logvar).item()

            mean_store = torch.cat((mean_store, mu), 0)

    all_data_mean = mean_store[1:]
    all_data_mean_np = all_data_mean.cpu().numpy()

    input_path = "C:/Users/mdzak/Desktop/GitHub/FYP_Zaki/results"
    input_path_name = input_path.split('/')[-1]
    latent_space_path = 'results/' + input_path_name + str(latent_dim) + 'D_latent_space_dna_meth.tsv'

    all_data_mean_df = pd.DataFrame(all_data_mean_np)
    # all_data_mean_df.to_csv(latent_space_path, sep='\t')

    print('=====> Average Test Loss: {:.3f}'.format(test_loss/len(dataloader.dataset)))

    return x_hat, data, z

# execute training and validation
print("Start Training")
for epoch in range(1, epochs + 1):
    train(model, full_loader, epoch)
    # x_hat, data, z = validate(model, test_loader, epoch)

# save trained model
torch.save(model.state_dict(), 'trained_models/VAE_dna_meth' + str(latent_dim) + '.pt')

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
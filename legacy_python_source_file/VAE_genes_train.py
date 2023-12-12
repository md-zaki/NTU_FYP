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
import VAE_genes_model

#========================= Training parameters ======================
epochs = 100
batch_size = 512
lr = 0.05
device = torch.device('cuda')

# ====================== Initialize the model ======================
input_size = 57820 #dimension of gene expressions
level_2 = 4096
level_3 = 2048
level_4 = 1024
latent_dim = 512 # target latent size

model = VAE_genes_model.VAE(input_size, level_2, level_3, level_4, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ====================== Prepare data =================================
genes = pd.read_csv('datasets_transpose_csv/genes_transpose.csv')
genes_ = genes.drop(columns=['Unnamed: 0'])
genes_ = genes_.drop(columns=['CELL_LINE'])
genes_np = genes_.to_numpy()
genes_train, genes_test = train_test_split(genes_np, test_size=0.10, random_state=42)

genes_train_t = torch.Tensor(genes_train)
genes_test_t = torch.Tensor(genes_test)
genes_full = torch.Tensor(genes_np)

# ===================== Define Dataloader =============================
train_loader = DataLoader(genes_train_t, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(genes_test_t,batch_size=batch_size, shuffle=False)
full_loader = DataLoader(genes_full,batch_size=batch_size, shuffle = False)

# ==================== Define Training Losses =========================
def recon_loss(x_hat, x): # Reconstruction Loss
        lossFunc = torch.nn.MSELoss()
        loss = lossFunc(x_hat, x)
        return loss

def kl_loss(mean, log_var): # KL Divergence
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss

# ================= Define Train Function ============================
def train(model, dataloader):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for batch_index, sample in enumerate(dataloader):
        data = sample
        data = data.to(device)

        optimizer.zero_grad()

        x_hat, mu, logvar = model(data)
        # x_hat = x_hat.to(device)

        train_recon_loss = recon_loss(x_hat, data)
        train_kl_loss = kl_loss(mu, logvar)
        loss = (train_recon_loss + train_kl_loss)
        total_recon += train_recon_loss.item()
        total_kl += train_kl_loss.item()
        total_loss +=loss.item()

        loss.backward()
        optimizer.step()
        
    train_loss = total_loss/len(dataloader.dataset)
    return train_loss, total_recon/len(dataloader.dataset), total_kl/len(dataloader.dataset)

# ===================== Define Validation Function ====================
def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_recon = 0.0
    with torch.no_grad():
        mean_store = torch.zeros(1, latent_dim).to(device)
        for batch_index, sample in enumerate(dataloader):
            data = sample
            data = data.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(data)

            val_recon_loss = recon_loss(x_hat, data)
            val_kl = kl_loss(mu, logvar)
            loss = (val_recon_loss + val_kl)
            
            total_kl += val_kl.item()
            total_recon+= val_recon_loss.item()
            total_loss +=loss.item()

            mean_store = torch.cat((mean_store, mu), 0)

    all_data_mean = mean_store[1:]
    all_data_mean_np = all_data_mean.cpu().numpy()

    input_path = "C:/Users/mdzak/Desktop/GitHub/FYP_Zaki/results"
    input_path_name = input_path.split('/')[-1]
    latent_space_path = 'results/' + input_path_name + str(latent_dim) + 'D_latent_space_gene_exp.tsv'

    all_data_mean_df = pd.DataFrame(all_data_mean_np)
    all_data_mean_df.to_csv(latent_space_path, sep='\t')

    val_avg_total_loss = total_loss/len(dataloader.dataset)
    val_avg_kl = total_kl/len(dataloader.dataset)
    val_avg_recon = total_recon/len(dataloader.dataset)

    return val_avg_total_loss, val_avg_kl, val_avg_recon

# ====================== Execute Training ========================

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_recons_loss, train_kl_loss = train(model, full_loader)
    # torch.cuda.empty_cache()
    val_epoch_loss, val_kl_loss, val_recon_loss = validate(model, full_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    # torch.cuda.empty_cache()
    print(f"Training Loss (KL plus MSE): {train_epoch_loss:.4f}")
    print(f"Training Loss (MSE): {train_recons_loss:.4f}")
    print(f"Training Loss (KL): {train_kl_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    print(f"Val KL Loss: {val_kl_loss:.4f}")
    print(f"Val Recon Loss (MSE): {val_recon_loss:.4f}")
    print()
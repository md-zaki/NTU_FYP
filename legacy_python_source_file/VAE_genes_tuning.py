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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
from functools import partial

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
# train_loader = DataLoader(genes_train_t, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(genes_test_t,batch_size=batch_size, shuffle=False)
# full_loader = DataLoader(genes_full,batch_size=batch_size, shuffle = False)

# ==================== Define Training Losses =========================
def recon_loss(x_hat, x): # Reconstruction Loss
        lossFunc = torch.nn.MSELoss()
        loss = lossFunc(x_hat, x)
        return loss

def kl_loss(mean, log_var): # KL Divergence
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss

# ================= Define Train Function for tuning ============================
def train( config, checkpoint_dir = None, full_loader = None):
    device = torch.device('cuda')
    model = VAE_genes_model.VAE(input_size,level_2, level_3, level_4, config['latent_dim']).to(device)

    

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    # full_loader = DataLoader(genes_train_t,batch_size=config['batch_size'], shuffle = False)

    for epoch in range(50):

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        epoch_steps = 0
        for batch_index, sample in enumerate(full_loader):
            data = sample
            data = data.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(data)
            x_hat = x_hat.to(device)

            train_recon_loss = recon_loss(x_hat, data)
            train_kl_loss = kl_loss(mu, logvar)
            loss = (train_recon_loss + train_kl_loss)
            total_recon += train_recon_loss.item()
            total_kl += train_kl_loss.item()
            total_loss +=loss.item()

            loss.backward()
            optimizer.step()
        
        
        train_loss = total_loss/len(full_loader.dataset)
        r_loss = total_recon/len(full_loader.dataset)
        k_loss = total_kl/len(full_loader.dataset)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=train_loss)
        print(f"Training Loss (KL plus MSE): {train_loss:.4f}")
        print(f"Training Loss (MSE): {r_loss:.4f}")
        print(f"Training Loss (KL): {k_loss:.4f}")
    print("Finished Training")



# ====================== Execute Tuning ========================

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    batch_size = 512

    full_loader = DataLoader(genes_train_t,batch_size=batch_size, shuffle = False)

    config = {
        # "level_2": tune.sample_from(lambda _: 2**np.random.randint(12,13)),
        # "level_3": tune.sample_from(lambda _: 2**np.random.randint(11, 12)),
        # "level_4": tune.sample_from(lambda _: 2**np.random.randint(10, 11)),
        "latent_dim": tune.grid_search([1024,512, 256, 128])
        # "batch_size": tune.choice([32,64,128,256,512]),
        # "lr": tune.loguniform(1e-4, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=max_num_epochs,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        tune.with_parameters(train,full_loader=full_loader),
        resources_per_trial={"cpu": 0, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    
    best_trained_model = VAE_genes_model.VAE(input_size,level_2, level_3, level_4, best_trial.config['latent_dim']).to(device)
    
    device = torch.device('cuda')
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)






if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, max_num_epochs=10, gpus_per_trial=1)
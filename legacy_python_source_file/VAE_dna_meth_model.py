import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_size, level_2, level_3, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.enc_fc1 = nn.Sequential(
                        nn.Linear(input_size, level_2),
                        nn.BatchNorm1d(level_2),
                        nn.ReLU())
        
        self.enc_fc2 = nn.Sequential(
                        nn.Linear(level_2, level_3),
                        nn.BatchNorm1d(level_3),
                        nn.ReLU())

        self.enc_fc3_mean = nn.Sequential(
                    nn.Linear(level_3, latent_dim),
                    nn.BatchNorm1d(latent_dim))
        
        self.enc_fc3_log_var = nn.Sequential(
                    nn.Linear(level_3, latent_dim),
                    nn.BatchNorm1d(latent_dim))
        
        
        # Decoder layers
        self.dec_fc3 = nn.Sequential(
                        nn.Linear(latent_dim, level_3),
                        nn.BatchNorm1d(level_3),
                        nn.ReLU())
        
        self.dec_fc2 = nn.Sequential(
                        nn.Linear(level_3, level_2),
                        nn.BatchNorm1d(level_2),
                        nn.ReLU())
        
        self.dec_fc1 = nn.Sequential(
                    nn.Linear(level_2, input_size),
                    nn.BatchNorm1d(input_size),
                    nn.Sigmoid())


    def encode(self, x):
        l2_layer = self.enc_fc1(x)
        l3_layer = self.enc_fc2(l2_layer)
        
        mu = self.enc_fc3_mean(l3_layer)
        logvar = self.enc_fc3_log_var(l3_layer)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        l3_layer = self.dec_fc3(z)
        l2_layer = self.dec_fc2(l3_layer)
        x_hat = self.dec_fc1(l2_layer)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z
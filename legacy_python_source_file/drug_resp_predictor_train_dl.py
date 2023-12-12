import drug_resp_predictor_model
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
from tqdm.notebook import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
import numpy as np
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import KFold
import time

# select dimension to train
selected_dim = 1024
selected_data = 'dna_meth'
# selected_data = 'gene_exp'

# import csv and merge with drugs
if selected_data == 'gene_exp':
    dimension = pd.read_csv('results/results' + str(selected_dim) + 'D_latent_space_' + str(selected_data) +'.tsv',sep='\t')
    cell_line_name = pd.read_csv('results_clean/cell_line_name.csv')
    gdsc_drug = pd.read_csv('results_clean/gdsc_drug_nodash.csv')
    dimension.drop(columns=['Unnamed: 0'], inplace=True)
    gdsc_drug.drop(columns=['Unnamed: 0'], inplace= True)
    dimension['CELL_LINE_NAME'] = cell_line_name['CELL_LINE_NAME']
    dimension_w_drug = pd.merge(dimension, gdsc_drug, on='CELL_LINE_NAME')
    dimension_w_drug.drop(columns=['CELL_LINE_NAME'],inplace=True)
elif selected_data == 'dna_meth':
    dimension = pd.read_csv('results/results' + str(selected_dim) + 'D_latent_space_' + str(selected_data) +'.tsv',sep='\t')
    cell_line_name = pd.read_csv('results_clean/cell_line_name_dna_meth.csv')
    gdsc_drug = pd.read_csv('results_clean/gdsc_drug_nodash.csv')
    dimension.drop(columns=['Unnamed: 0'], inplace=True)
    gdsc_drug.drop(columns=['Unnamed: 0'], inplace= True)
    dimension['CELL_LINE_NAME'] = cell_line_name['CELL_LINE']
    dimension_w_drug = pd.merge(dimension, gdsc_drug, on='CELL_LINE_NAME')
    dimension_w_drug.drop(columns=['CELL_LINE_NAME'],inplace=True)

#shuffle data
dimension_w_drug = dimension_w_drug.sample(frac=1, random_state=33).reset_index(drop=True)

#seperate continuous and categorical columns
cat_cols = ['DRUG_NAME']
cont_cols = dimension_w_drug.drop(columns=['DRUG_NAME', 'LN_IC50']).columns
label_cols = ['LN_IC50']

# Category to tensor
for cat in cat_cols:
    dimension_w_drug[cat] = dimension_w_drug[cat].astype('category')
drug_name = dimension_w_drug['DRUG_NAME'].cat.codes.values
cats = np.stack([drug_name],1)
cats = torch.tensor(cats, dtype=torch.int64)

# Cont to tensor
conts = np.stack([dimension_w_drug[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)

# Labels to tensor
labels = np.stack([dimension_w_drug[col].values for col in label_cols], 1)
labels = torch.tensor(labels, dtype=torch.float)

# Set embedding size for categorical columns
cat_szs = [len(dimension_w_drug[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

# Seed value and device
torch.manual_seed(33)
device = torch.device('cuda')

# Declare Model
model = drug_resp_predictor_model.TabularModel(emb_szs, conts.shape[1], 1, [200,100], p=0.4).to(device)

# Declare loss and optimzer function
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test splits
cat_train, cat_test, y_train, y_test, con_train, con_test = train_test_split(cats, labels, conts, test_size=0.2, random_state= 0, shuffle=False)

train_dataset = torch.utils.data.TensorDataset(cat_train, con_train, y_train)
test_dataset = torch.utils.data.TensorDataset(cat_test, con_test, y_test)

# Declare dataloaders
batch_size = 256
train_loader = DataLoader(train_dataset, shuffle=False, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

# Train loop
start_time = time.time()

epochs = 100
losses = []

print('Selected data: ' + str(selected_data))
print("Training with dimension size: " + str(con_train.shape[1]))
for i in range(epochs):
    i+=1
    best_loss = np.inf
    for cat_train, con_train, y_train in train_loader:
        cat_train = cat_train.to(device)
        con_train = con_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(cat_train, con_train)
        y_pred = y_pred.to(device)
        loss = torch.sqrt(loss_fn(y_pred, y_train))

        if loss.item() < best_loss:
            best_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i%10 == 1:
        print(f'epoch: {i:3}  training loss (rmse): {best_loss:10.8f}')
    losses.append(best_loss)

    

print(f'epoch: {i:3}  loss: {best_loss:10.8f}')
print(f'\nTraining Time: {time.time() - start_time:.0f} seconds')

# Plot loss function
# plt.plot(range(epochs), losses)
# plt.ylabel('RMSE Loss')
# plt.xlabel('epoch')
# plt.show()

# Validate model using test set
with torch.no_grad():
    i = 0
    for cat_test, con_test, y_test in test_loader:
        cat_test = cat_test.to(device)
        y_test = y_test.to(device)
        con_test = con_test.to(device)
        
        y_val = model(cat_test, con_test)
        loss = torch.sqrt(loss_fn(y_val, y_test))
        i += 1
        print(i)
print(f'Validation RMSE: {loss:.8f}')

# See sample of predicted, actual and difference
# print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
# for i in range(50):
#     diff = np.abs(y_val[i].item()-y_test[i].item())
#     print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')
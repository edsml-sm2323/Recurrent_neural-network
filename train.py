import torch
import torch.nn as nn
from model import RNNModel, LSTM, GRU
import torch.optim as optim
from torch.utils.data import Subset
from dataloader import GoldPriceDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd



# set device function
def set_device(device="cpu", idx=0):
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print("Cuda installed! Running on GPU {} {}!".format(idx, torch.cuda.get_device_name(idx)))
            device="cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print("Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(torch.cuda.device_count(), torch.cuda.get_device_name()))
            device="cuda:0"
        else:
            device="cpu"
            print("No GPU available! Running on CPU")
    return device

device = set_device()
model = GRU().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# dataloader
df_lstm = pd.read_csv("daily_gold_rate.csv")
drop = ['EUR', 'USD','INR' , 'AED', 'CNY','Date']
df = df_lstm.drop(drop, axis = 1)

# Normal split
dataset = GoldPriceDataset(df,30)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.9)
test_size = dataset_size - train_size

# Split train and test dataset
train_dataset = Subset(dataset, range(0, train_size))
test_dataset = Subset(dataset, range(train_size, dataset_size))

#Split train and validation dataset
train_size = len(train_dataset)
train_size2 = int(train_size * 0.9)
val_size = train_size - train_size2

train_dataset1 = Subset(train_dataset, range(0, train_size2))
val_dataset = Subset(train_dataset, range(train_size2, train_size))

# Create dataloader
train_loader = DataLoader(train_dataset1, batch_size=10, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



# Training
best_loss = float('inf')
num_epochs = 50
for epoch in range(50):
    model.train()
    for tx, ty in train_loader:
        output = model(torch.unsqueeze(tx, dim=2))
        loss = criterion(torch.squeeze(output), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}')

    model.eval()
    for tx, ty in val_loader:
        output = model(torch.unsqueeze(tx, dim=2))
        val_loss = criterion(torch.squeeze(output), ty)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss.item():.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
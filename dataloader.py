import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class GoldPriceDataset(Dataset):
    def __init__(self, dataframe, n_days):
        self.n_days = n_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(dataframe.iloc[:, 0].values.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.data) - self.n_days

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.n_days]
        y = self.data[idx+self.n_days]
        return torch.from_numpy(x).float(), torch.tensor(y).float()
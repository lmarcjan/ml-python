
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from util.df_util import load, drop
from util.stat_util import predict_error

class ModelDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    housing_df = load('housing.csv')
    train, test, = train_test_split(housing_df, random_state=42)
    train_X = drop(train, ["median_house_value"]).fillna(0)
    train_y = train["median_house_value"]
    test_X = drop(test, ["median_house_value"]).fillna(0)
    test_y = test["median_house_value"]
    m, n = train_X.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X_tensor = torch.tensor(train_X.to_numpy()).float().to(device)
    train_y_tensor = torch.tensor(train_y.to_numpy()).float().reshape(-1, 1).to(device)

    ds = ModelDataset(train_X_tensor, train_y_tensor)
    dataloader = DataLoader(ds, batch_size=10, shuffle=True)

    model = nn.Sequential(
        nn.Linear(n, int(n/2)),
        nn.ReLU(),
        nn.Linear(int(n/2), 1),
        nn.ReLU()
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    n_epochs = 30
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        for data in dataloader:
            x, y = data
            optimizer.zero_grad()
            loss_val = loss_fn(model(x), y)
            loss_val.backward()
            optimizer.step()

    predict_error(model(train_X_tensor).detach().cpu().numpy(), train_y, "Train")
    test_X_tensor = torch.tensor(test_X.to_numpy()).float().to(device)
    predict_error(model(test_X_tensor).detach().cpu().numpy(), test_y, "Test")

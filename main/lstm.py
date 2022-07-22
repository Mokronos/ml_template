import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

# set random seed
torch.manual_seed(3)

class BasicDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


# X 1s --> y time dependent but linear
# y = np.array(list(range(1,10)))
# X = np.array([1]*len(y))
# y = y.reshape((1,len(y),1))
# X = X.reshape((1,len(X),1))

X = np.array(list(range(1,10)))
X = X.reshape((1,len(X),1))
y = X

cutoff = int(X.shape[1] * 2/3 )
X_train, y_train = X[:,:cutoff], y[:,:cutoff]
X_test, y_test = X[:,cutoff:], y[:,cutoff:]
# X_train, y_train = X[:,:6], y[:,:6]
# X_test, y_test = X[:,:6], y[:,:6]
print(X_train)

training_data = BasicDataset(X_train, y_train)
test_data = BasicDataset(X_test, y_test)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        # self.fc.weight.data.fill_(1)
        # self.fc.bias.data.fill_(0)

    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss.item())
        print(pred)

        # for weight in model.named_parameters():
        #     print(f"{weight}")
        print(f"loss: {loss.item()}")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(X,y)
            print(pred)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork(1,5,1).to(device)
print(model)

epochs = 1000
lr = 1e-1

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_vals = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
test(test_dataloader, model, loss_fn)
x = torch.tensor([[93]], dtype=torch.float32, device="cuda")
print(x)
print(model(x))
print("Done!")
plt.plot(loss_vals)
plt.show()

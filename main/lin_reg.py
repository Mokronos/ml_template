import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


# linear mapping
X = np.array(list(range(1,10)))
X = X.reshape((len(X),1))
y = X

cutoff = int(X.shape[0] * 2/3 )
X_train, y_train = X[:cutoff], y[:cutoff]
X_test, y_test = X[cutoff:], y[cutoff:]

training_data = BasicDataset(X_train, y_train)
test_data = BasicDataset(X_test, y_test)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(X.type())
    print(X)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size,1)
        # self.fc.weight.data.fill_(1)
        # self.fc.bias.data.fill_(0)

    def forward(self, x):
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

        for weight in model.named_parameters():
            print(f"{weight}")
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

model = NeuralNetwork(1).to(device)
print(model)

epochs = 100
lr = 1e-2

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
test(test_dataloader, model, loss_fn)
x = torch.tensor([[93]], dtype=torch.float32, device="cuda")
print(x)
print(model(x))
print("Done!")

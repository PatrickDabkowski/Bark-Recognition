import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
print('ANN Model \n')

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} \n")


if __name__ == '__main__':
    X = np.load('X.npy')
    y = np.load('y.npy')
    y = np.array([0 if l == 'cat' else 1 for l in y])
    # cat is 0, dog is 1
    print('X shape: ', X.shape, '\ny shape: ', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=12)
    test_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))


    class ANN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.input = nn.Linear(in_features=input_size, out_features=2 * input_size).to(device)
            self.hidden_1 = nn.Linear(in_features=2 * input_size, out_features=4 * input_size).to(device)
            self.hidden_2 = nn.Linear(in_features=4 * input_size, out_features=4 * input_size).to(device)
            self.output = nn.Linear(in_features=4 * input_size, out_features=2).to(device)

        def forward(self, x):
            x = nn.functional.relu(self.input(x))
            x = nn.functional.relu(self.hidden_1(x))
            x = nn.functional.relu(self.hidden_2(x))
            return torch.sigmoid(self.output(x))

    model = ANN(X.shape[1])
    print(model)

    num_epochs = 200
    train_accuracies, test_accuracies = [], []

    loss = nn.CrossEntropyLoss()
    adam = torch.optim.Adam(params=model.parameters(), lr=0.01)

    for epoch in range(num_epochs):

        # Train set
        for X, y in train_loader:
            preds = model(X.to(device))
            pred_labels = torch.argmax(preds, axis=1)
            loss_ = loss(preds, y.long())
            print('epoch: ', epoch, ' Loss: ', loss_)
            adam.zero_grad()
            loss_.backward()
            adam.step()
        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        # Test set
        X, y = next(iter(test_loader))
        pred_labels = torch.argmax(model(X), axis=1)
        test_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(test_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")

    fig.align_labels()
    plt.show()


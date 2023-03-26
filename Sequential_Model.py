import torch
import torch.nn as nn
import numpy as np
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
print('CUDA available:', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} \n")


if __name__ == '__main__':
    X = np.load('X.npy')
    y = np.load('y.npy')
    y = np.array([0 if l == 'cat' else 1 for l in y])
    # cat is 0, dog is 1
    print('X shape: ', X.shape, '\ny shape: ', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # Multithread processing
    train_loader = DataLoader(train_data, shuffle=True, batch_size=10, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))

    model = nn.Sequential(
        nn.Linear(8800, 4400),
        nn.ReLU(),
        nn.Linear(4400, 2200),
        nn.ReLU(),
        nn.Linear(2200, 1100),
        nn.ReLU(),
        nn.Linear(1100, 2),
        nn.LogSoftmax(dim=1))

    print(model)

    num_epochs = 200
    train_accuracies, test_accuracies = [], []

    loss = nn.CrossEntropyLoss()
    adam = torch.optim.RMSprop(params=model.parameters(), lr=0.001)

    best_test_loss = 0
    patience = 25
    count = 0

    best_model = model
    best_accuracy = 0

    for epoch in range(num_epochs):

        # Train set
        batch = 0
        for X, y in train_loader:
            preds = model(X.to(device))
            pred_labels = torch.argmax(preds, axis=1)
            loss_ = loss(preds, y.long())

            print('Batch: ', batch, ' Loss: ', loss_)
            adam.zero_grad()
            loss_.backward()
            adam.step()
            batch += 1

        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        test_preds = []
        test_labels = []
        for X, y in test_loader:
            preds = model(X)
            pred_labels = torch.argmax(preds, axis=1)
            test_preds.extend(pred_labels.numpy())
            test_labels.extend(y.numpy())
        test_accuracy = 100 * np.mean(np.array(test_preds) == np.array(test_labels))
        print('epoch: ', epoch, ' Accuracy: ', 100 * torch.mean((pred_labels == y).float()).item())

        if test_accuracy > best_accuracy:
            best_model = model

        if test_accuracy > best_test_loss:
            best_test_loss = test_accuracy
            count = 0
        else:
            count += 1
            if count >= patience:
                print('Early stopping...')
                break

        # Test set
        X, y = next(iter(test_loader))
        pred_labels = torch.argmax(model(X), axis=1)
        test_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

    torch.save(best_model.state_dict(), 'Sequential_Model_Torch')

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
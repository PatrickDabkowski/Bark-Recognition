import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torch.nn as nn
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA')
    else:
        print('CPU')
        device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((150, 100)),
        transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='/content/drive/My Drive/Dataset', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print('Train size: ', train_size)
    print('Test size: ', test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 128, shuffle=True, num_workers= 64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 128, shuffle=True, num_workers= 64)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, padding='valid'),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, padding='valid'),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.2),
        nn.Conv2d(32, 64, 3, padding='valid'),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding='valid'),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(p=0.4),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(128, 256, 3, padding='valid'),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        nn.Linear(in_features= 28672, out_features=3),
        nn.Softmax(dim=1))

    model = model.to(device)
    print(model)

    num_epochs = 100000
    train_accuracies, test_accuracies = [], []

    loss = nn.CrossEntropyLoss()
    adam = torch.optim.Adam(params=model.parameters(), lr=0.002)

    best_test_loss = 0
    patience = 55
    count = 0

    best_model = model
    best_accuracy = 0

    for epoch in range(num_epochs):
        # Train set
        batch = 0
        for X, y in train_loader:
            X, y = X.to(device='cuda'), y.to(device='cuda')
            y = torch.squeeze(y)
            preds = model(X)
            pred_labels = torch.argmax(preds, axis=1)
            loss_ = loss(preds, y.long())
            print('Batch: ', batch, ' Loss: ', loss_)
            loss_.backward()
            adam.step()
            batch += 1

        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        test_correct = 0
        test_total = 0
        with torch.no_grad():
          for X, y in test_loader:
            X, y = X.to(device='cuda'), y.to(device='cuda')
            preds = model(X).to(device='cuda')
            pred_labels = torch.argmax(preds, axis=1)
            test_correct += (pred_labels == y).sum().item()
            test_total += y.size(0)

        test_accuracy = 100 * test_correct / test_total

        print('epoch: ', epoch, ' Accuracy: ', test_accuracy)

        if test_accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = test_accuracy

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
        X, y = X.to(device='cuda'), y.to(device='cuda')
        test_preds = torch.argmax(model(X), axis=1)
        test_preds = test_preds.cpu().numpy()  # Skopiowanie tensora na CPU
        test_accuracy = 100 * np.mean(np.array_equal(test_preds, y.cpu().numpy()))
        test_accuracies.append(test_accuracy)

    print('Best Accuracy: ', best_accuracy)
    torch.save(best_model, '/content/drive/My Drive/Sciezka/Do/Folderu/Image_Model_Torch')




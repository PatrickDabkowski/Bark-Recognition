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
        transforms.Resize((128, 128)),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.01),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='/content/drive/My Drive/Dataset', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print('Train size: ', train_size)
    print('Test size: ', test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 64, shuffle=True, num_workers= 32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 64, shuffle=True, num_workers= 32)

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding='valid'),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding='valid'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.1),
        nn.Conv2d(64, 128, 3, padding='valid'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.2),
        nn.Conv2d(128, 256, 3, padding='valid'),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Dropout(p=0.2),
        nn.Flatten(),
        nn.Linear(in_features= 9216, out_features=4608),
        nn.ReLU(),
        nn.Linear(in_features= 4608, out_features=1),
        nn.Sigmoid())

    model = model.to(device)
    print(model)

    num_epochs = 200
    train_accuracies, test_accuracies = [], []

    loss = nn.BCELoss()
    adam = torch.optim.Adam(params=model.parameters(), lr=0.005)

    patience = 20
    count = 0
    best_model = model
    best_accuracy = 0

    for epoch in range(num_epochs):
        # Train set
        batch = 0
        for X, y in train_loader:
          adam.zero_grad()
          X, y = X.to(device='cuda'), y.to(device='cuda')
          y = y.float()
          preds = model(X)
          pred_labels = (preds > 0.5).float() 
          loss_ = loss(preds, y.unsqueeze(1)) 
          print('Batch: ', batch, ' Loss: ', loss_)
          loss_.backward()
          adam.step()
          batch += 1
          train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        # Test set
        X, y = next(iter(test_loader))
        X, y = X.to(device='cuda'), y.to(device='cuda')
        test_preds = torch.argmax(model(X), axis=1)
        test_preds = test_preds.cpu().numpy() 
        test_accuracy = 100 * torch.mean((pred_labels == y).float()).item()
        test_accuracies.append(test_accuracy)
        print('epoch: ', epoch, ' Accuracy: ', test_accuracy)

        if test_accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = test_accuracy
        else:
            count += 1
            if count >= patience:
                print('Early stopping...')
                break

    print('Best Accuracy: ', best_accuracy)
    torch.save(best_model, '/content/drive/My Drive/Sciezka/Do/Folderu/Image_Model_Torch')


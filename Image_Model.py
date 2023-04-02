import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Cores: ', os.cpu_count())

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.Grayscale(),
        transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='kagglecatsanddogs_5340/PetImages', transform=transform)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    print('Train size: ', train_size)
    print('Test size: ', test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 20, shuffle=True, num_workers= os.cpu_count())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 20, shuffle=True, num_workers= os.cpu_count())

    model = nn.Sequential(
        #nn.Conv2d(3, 3, 10, padding='valid'),
        #nn.ReLU(),
        nn.Conv2d(1, 3, 5, padding='valid'),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Conv2d(3, 3, 3, padding='valid'),
        nn.ReLU(),
        nn.MaxPool2d(4, stride=2),
        #nn.Conv2d(3, 3, 5, padding='valid'),
        #nn.ReLU(),
        nn.Conv2d(3, 3, 3, padding='valid'),
        nn.ReLU(),
        #nn.Conv2d(3, 3, 3, padding='valid'),
        #nn.ReLU(),
        nn.MaxPool2d(4, stride=2),
        nn.Conv2d(3, 2, 2, stride=2, padding='valid'),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        nn.Linear(in_features=1682, out_features=2),
        nn.Sigmoid())

    print(model)

    num_epochs = 200
    train_accuracies, test_accuracies = [], []

    loss = nn.CrossEntropyLoss()
    adam = torch.optim.RMSprop(params=model.parameters(), lr=0.1)

    best_test_loss = 0
    patience = 10
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
            loss_.backward()
            adam.step()
            batch += 1

        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        test_preds = []
        test_labels = []
        for X, y in test_loader:
            preds = model(X)
            preds = preds.squeeze()
            test_preds.extend((preds > 0.5).np())
            test_labels.extend(y.np())
        test_accuracy = 100 * np.mean(np.array(test_preds) == np.array(test_labels))
        print('epoch: ', epoch, ' Accuracy: ', 100 * torch.mean((pred_labels == y).float()).item())

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
        pred_labels = torch.argmax(model(X), axis=1)
        test_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

    print('Best Accuracy: ', best_accuracy)
    torch.save(best_model, 'Image_Model_Torch')

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





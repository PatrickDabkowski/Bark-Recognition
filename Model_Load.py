import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X = np.load('X.npy')
y = np.load('y.npy')
y = np.array([0 if l == 'cat' else 1 for l in y])
# cat is 0, dog is 1
print('X shape: ', X.shape, '\ny shape: ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
# Multithread processing

test_loader = DataLoader(test_data)

model = nn.Sequential(
    nn.Linear(8800, 4400),
    nn.ReLU(),
    nn.Linear(4400, 2200),
    nn.ReLU(),
    nn.Linear(2200, 1100),
    nn.ReLU(),
    nn.Linear(1100, 2),
    nn.LogSoftmax(dim=1))

model.load_state_dict(torch.load('Sequential_Model_Torch'))

# Oblicz dokładność modelu na zestawie testowym
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Model Accuracy: ', accuracy)

'''
X shape:  (599, 8800) 
y shape:  (599,)
Model Accuracy:  94.16666666666667'''
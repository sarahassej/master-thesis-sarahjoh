import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CustomDatasetOneD(Dataset):
    def __init__(self, x, y, num_cl, transform=None, target_transform=None):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        y = y - 1
        y = torch.from_numpy(y)

        self.data = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.num_cl = num_cl

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx, :, :]
        label = self.labels[idx]
        one_hot = F.one_hot(label, num_classes=self.num_cl).float()
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {'x': x, 'label': label, 'one_hot': one_hot}
        return sample

class CNN_ONEDIM(nn.Module):
    def __init__(self):
        super(CNN_ONEDIM, self).__init__()

        self.cnn_layer1 = nn.Sequential(nn.Conv1d(1, 256, kernel_size=5, stride=1, padding='same'),
                                        nn.BatchNorm1d(256), nn.ReLU())
        self.cnn_layer2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=5, stride=1, padding='same'), nn.ReLU(),
                                        nn.Dropout1d(p=0.1), nn.BatchNorm1d(128), nn.MaxPool1d(8))
        self.cnn_layer3 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same'), nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same'), nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same'),
                                        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout1d(p=0.2))
        self.cnn_layer4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same'), nn.ReLU(),
                                        nn.Flatten(), nn.Dropout(p=0.2))
        self.cnn_layers = nn.Sequential(self.cnn_layer1, self.cnn_layer2, self.cnn_layer3, self.cnn_layer4)

        self.fc_layer = nn.Sequential(nn.LazyLinear(6), nn.BatchNorm1d(6), nn.Softmax(dim=1))  # 11648

    # Forward-function passing the input thru the layers, and returning a one-dimensional tensor of the softmax
    # probabilities.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc_layer(x)
        return x
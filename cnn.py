import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score
from pytorchtools import EarlyStopping


class CustomDataset(Dataset):
    def __init__(self, x, y, num_cl, transform=None, target_transform=None):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
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
        x = self.data[idx, :, :, :]
        label = self.labels[idx]
        one_hot = F.one_hot(label, num_classes=self.num_cl).float()
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {'x': x, 'label': label, 'one_hot': one_hot}
        return sample


class CNN(nn.Module):

    def __init__(self, dropout1=0.1, dropout2=0.2, dropout3=0.2):
        super(CNN, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1)),
                                        nn.BatchNorm2d(256), nn.ReLU())
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1)), nn.ReLU(),
                                        nn.Dropout2d(dropout1), nn.BatchNorm2d(128), nn.MaxPool2d(8))
        self.cnn_layer3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)), nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)), nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
                                        nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout2))
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
                                        nn.Flatten(), nn.Dropout(dropout3))
        self.cnn_layers = nn.Sequential(self.cnn_layer1, self.cnn_layer2, self.cnn_layer3, self.cnn_layer4)

        self.fc_layer = nn.Sequential(nn.LazyLinear(6), nn.BatchNorm1d(6), nn.Softmax(dim=1))  # 11648

    # Forward-function passing the input thru the layers, and returning a one-dimensional tensor of the softmax
    # probabilities.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc_layer(x)
        return x


def train_epoch(model, dataloader_train, optimizer, criterion, device):
    model.train()
    losses = []
    train_correct = 0

    for batch_idx, data in enumerate(dataloader_train):
        inputs = data['x']
        labels = data['one_hot']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction
        outputs = model(inputs)

        # computing the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        scores, predictions = torch.max(outputs.data, 1)
        labs = torch.argmax(labels, 1)
        train_correct += (predictions == labs).sum().item()

    return np.mean(losses), train_correct


def evaluate_epoch(model, dataloader, criterion, num_cl, device):
    concat_pred = [np.empty(shape=0) for _ in range(num_cl)]
    # prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = [np.empty(shape=0) for _ in range(num_cl)]
    avgprecs = np.zeros(num_cl)

    model.eval()
    losses = []
    val_correct = 0

    with torch.no_grad():

        for batch_idx, data in enumerate(dataloader):
            inputs = data['x']
            labels = data['one_hot']

            inputs = inputs.to(device)
            labels = labels.to(device)

            # prediction
            outputs = model(inputs)

            # computing the loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            out = outputs.to('cpu')
            lab = labels.to('cpu')

            scores, predictions = torch.max(out.data, 1)
            labs = torch.argmax(lab, 1)
            val_correct += (predictions == labs).sum().item()

            concat_pred = np.concatenate((concat_pred, np.transpose(out)), axis=1)
            concat_labels = np.concatenate((concat_labels, np.transpose(lab.numpy())), axis=1)

    for c in range(num_cl):
        avgprecs[c] = average_precision_score(concat_labels[c], concat_pred[c])

    return avgprecs, np.mean(losses), val_correct, concat_labels, concat_pred


def train_and_evaluate(epochs, model, dataloader_train, dataloader_test, optimizer, criterion,
                       device, patience, num_cl=6, scheduler=None, output=None, path=''):
    weights = None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_perfs': [], 'best_epoch': epochs}

    if output:
        trace_func = output.write
    else:
        trace_func = print
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path+'checkpoint.pt', trace_func=trace_func)

    for epoch in range(epochs):
        if output:
            output.write('Epoch {}/{}\n'.format(epoch + 1, epochs))
            output.write('-' * 10 + '\n')
        else:
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)
        start = time.time()
        train_loss, train_correct = train_epoch(model, dataloader_train, optimizer, criterion, device)
        end = time.time() - start

        if scheduler:
            scheduler.step()

        avgprecs, val_loss, val_correct, concat_labels, concat_pred = evaluate_epoch(model, dataloader_test,
                                                                                     criterion, num_cl, device)
        avgperfmeasure = np.mean(avgprecs)
        if output:
            output.write('at epoch: ' + str(epoch + 1) + '\n')
            output.write(' classwise perfmeasure ' + str(avgprecs) + '\n')
            output.write(' avgperfmeasure ' + str(avgperfmeasure) + '\n')
            output.write(' time ' + str(end) + '\n')
            output.write('-' * 5 + '\n\n')
        else:
            print('at epoch: ', epoch+1)
            print(' classwise perfmeasure ', avgprecs)
            print(' avgperfmeasure ', avgperfmeasure)
            print(' time ', end)
            print('-' * 5)

        train_loss = train_loss / len(dataloader_train.sampler)
        train_acc = train_correct / len(dataloader_train.sampler) * 100
        val_loss = val_loss / len(dataloader_test.sampler)
        val_acc = val_correct / len(dataloader_test.sampler) * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_perfs'].append(avgprecs)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            if output:
                output.write("Early stopping")
            else:
                print("Early stopping")
            history['best_epoch'] = epoch - patience
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path+'checkpoint.pt'))
    weights = deepcopy(model.state_dict())

    return history, weights

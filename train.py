import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader

class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs):
    plt.plot(train_acc_list)
    plt.ylabel('Train Accuracy')
    plt.savefig('Train_Accuracy.png')
    plt.show()

def plot_train_loss(train_loss_list, epochs):
    plt.plot(train_loss_list)
    plt.ylabel('Train Loss')
    plt.savefig('Train_Loss.png')
    plt.show()

def plot_test_acc(test_acc_list, epochs):
    plt.plot(test_acc_list)
    plt.ylabel('Test Accuracy')
    plt.savefig('Test_Accuracy.png')
    plt.show()

def train(model, loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    torch.save(best_wts, './weights/best.pt')
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc

in_alpha = 0.1
in_p = 0.6

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 750

        # Layer 1
        self.conv1 = nn.Conv2d(1,16,kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)

        # Layer 2
        self.conv2 = nn.Conv2d(16,32,kernel_size=(2,5),stride=(1,1),groups=16,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        self.elu2 = nn.ELU(alpha=in_alpha)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0)
        self.dropout2 = nn.Dropout(p=in_p)

        # Layer 3
        self.conv3 = nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        self.elu3 = nn.ELU(alpha=in_alpha)
        self.avgpool3 = nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0)
        self.dropout3 = nn.Dropout(p=in_p)
        
        # FC
        self.linear4 = nn.Linear(in_features=736,out_features=2,bias=True)


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.avgpool3(x)
        x = self.dropout3(x)

        # FC
        x = x.view(-1,736)
        x = self.linear4(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-lr", type=float, default=0.005)
    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EEGNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)
    summary(model, (1, 2, 750))

train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, args)

print(max(test_acc_list))
plot_train_acc(train_acc_list, args.num_epochs)
plot_train_loss(train_loss_list, args.num_epochs)
plot_test_acc(test_acc_list, args.num_epochs)
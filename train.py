from sched import scheduler
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.dataset import diy_Dataset
import random
from torch import nn
from utils.tools import train, validate
import torchvision
from torchvision import models
import pandas as pd
import wandb

wandb.init(project="death-cls", name="resnet50")

device = torch.device('cuda:0')

print("Loading data...")

data = diy_Dataset('data/matrix.mat', 'data/gt.mat')

idx = int(len(data) * 0.25)

batch_size = 8

traindata, testdata = random_split(data, [len(data) - idx, idx])

train_loader = DataLoader(traindata, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(testdata, batch_size=batch_size, drop_last=True, shuffle=True)

print(f'Complete!\nNum of train set: {len(traindata)}\nNum of val set: {len(testdata)}')

net = nn.Sequential(
    nn.Conv2d(1, 3, 1, padding=0, stride=1), models.resnet50(pretrained=True), nn.Linear(1000, 2)
)

net = net.to(device)

acc = validate(test_loader, net, device)
print(f'Initial acc: {100*acc:8f}%')

loss = nn.CrossEntropyLoss()
updater = torch.optim.AdamW(net.parameters(), lr=1e-4)
# updater = torch.optim.SGD(net.parameters(), lr=0.0025)

torch.cuda.empty_cache()

num_epochs = 300
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(updater, mode='min', patience=5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(updater, num_epochs)

wandb.watch(net, log="all")

train(net=net, train_iter=train_loader, test_iter=test_loader, loss=loss, updater=updater, num_epochs=num_epochs, device=device, scheduler=scheduler, save=True)

from asyncore import write
import imp
import torch
from tqdm import tqdm
import numpy as np
import datetime
import os
import wandb
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


def validate(test_iter, net, device):
    net.eval()
    accumulate = torch.tensor([0, 0]).float()
    print('starting val...')
    with torch.no_grad():
        for X, y in tqdm(test_iter):
            flag = 0
            X, y = X.to(device), y.to(device)
            pred = F.softmax(net(X), dim=-1)
            cmp = torch.squeeze(torch.argmax(pred, dim=-1)) == y
            flag = np.float(cmp.type(X.dtype).sum())
            accumulate += torch.tensor([flag, X.shape[0]]).float()
        print(accumulate)
    return accumulate[0] / accumulate[1]

def train_epoch(net, train_iter, loss, updater, device, scheduler):
    net.train()
    print('starting train...')
    for X, y in tqdm(train_iter):
        X, y = X.to(device), y.to(device)
        y_pred = net(X)
        l = loss(torch.squeeze(y_pred), y).mean()
        updater.zero_grad()
        l.backward(retain_graph=True)
        # l.backward()
        updater.step()

    # scheduler.step(l.item())
    scheduler.step()

    return l.item()

def train(net, train_iter, test_iter, loss, updater, num_epochs, device, scheduler, save=True):
    acc = 0
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join('.', 'log', 'train_log_' + time_str)
    writer = SummaryWriter(log_dir)
    for epoch in range(num_epochs):
        print('\nStarting Epoch: ', f'{epoch+1}...\n')
        l = train_epoch(net, train_iter, loss, updater, device, scheduler)
        acc_new = validate(test_iter, net, device)

        if save:
            if acc_new > acc:
                os.makedirs('./ckpt', exist_ok=True)
                torch.save(net.state_dict(), './ckpt/best.pth')
                acc = acc_new

            if (epoch+1) % 5 == 0:
                torch.save(net.state_dict(), f'./ckpt/ckpt_epoch_{epoch+1}.pth')


        train_acc = validate(train_iter, net, device)

        lr = updater.param_groups[0]['lr']
        print(f'\nTrain Acc: {100*train_acc:.8f}%\nTest Acc: {100*acc_new:.8f}%\nLoss: {l:.8f}\nLearning rate: {lr:.8f}')
        writer.add_scalar('Loss', l, epoch)
        writer.add_scalar('acc/Valid Accuracy', acc_new, epoch)
        writer.add_scalar('acc/Train Accuracy', train_acc, epoch)
        writer.add_scalar('Learning Rate', updater.param_groups[0]['lr'], epoch)

        wandb.log({
            'Valid Accuracy': acc_new,
            'Train Accuracy': train_acc,
            'loss': l,
            'Learning Rate': updater.param_groups[0]['lr']
            })

    writer.close()
    torch.save(net.state_dict(), f'./ckpt/final.pth')

def predict(sample, net, device):
    net = net.to(device)
    sample = torch.tensor(sample).to(device)

    X = net(sample)
    predict = torch.argmax(X)

    return predict
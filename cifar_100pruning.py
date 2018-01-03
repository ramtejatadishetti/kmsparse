from __future__ import print_function


import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random

import torchvision
import os
import torchvision.transforms as transforms
#from utils import progress_bar
from custom import *
import argparse

random.seed(1)




class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        #define the layers
        self.conv1 = NewMaskedConv2D(3, 64, 5, 1)
        self.conv2 = NewMaskedConv2D(64, 64, 5, 1)
        self.conv3 = NewMaskedConv2D(64, 64, 5, 1)

        #self.conv3 = NewMaskedConv2D(64, 128, 5, 1)

        #self.fc1 = NewMaskedLayer(1024, 384)
        #self.fc2 = NewMaskedLayer(384, 192)
        self.fc3 = NewMaskedLayer(1024, 100)
        #print(self.conv1.weight.data.size())

    def forward(self, x, update_flag):

        #out = x
        if update_flag == True:

            out = F.relu(self.conv1(x, True))
            out = F.max_pool2d(out, 3, stride=2)
        
            out = F.relu(self.conv2(out, True))
            out = F.relu(self.conv3(out, True))
            out = F.max_pool2d(out, 3, stride=2)

#            out = F.relu(self.conv3(out, True))
#            out = F.max_pool2d(out, 3, stride=2)

            out = out.view(out.size(0), -1)
        
        #print(out.shape)
            #out = F.relu(self.fc1(out, True))
            #out = F.relu(self.fc2(out, True))
            out = F.softmax(self.fc3(out, True))

        else:
            out = F.relu(self.conv1(x, False))
            out = F.max_pool2d(out, 3, stride=2)
        
            out = F.relu(self.conv2(out, False))
            out = F.relu(self.conv3(out, False))
            out = F.max_pool2d(out, 3, stride=2)


 #           out = F.relu(self.conv3(out, False))
 #           out = F.max_pool2d(out, 3, stride=2)

            out = out.view(out.size(0), -1)
        
        #print(out.shape)
            #out = F.relu(self.fc1(out, False))
            #out = F.relu(self.fc2(out, False))
            out = F.softmax(self.fc3(out, False))




        return out



if __name__ == "__main__":


    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #bd_layer = BinaryLayer(3, 4, 1, True)i

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store_true", dest='use_cuda',  default=False)
    parser.add_argument('-e', action="store", dest="epoch_count", type=int, default= 100)
    parser.add_argument('-b', action="store", dest="prune_start", type=int, default= 0)
    parser.add_argument('-p', action="store", dest="prune_count", type=int, default=0)
    FLAGS = vars(parser.parse_args())
    print(FLAGS)

    use_cuda = FLAGS['use_cuda']
    epoch_count = FLAGS['epoch_count']
    prune_count = FLAGS['prune_count']
    prune_start = FLAGS['prune_start']

    net = MyNetwork()
    print(net)

    if use_cuda:
        net = net.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    start_epoch = 0
    end_epoch = epoch_count
    best_acc = 0

    count = 0
    flag = 1

    for epoch in range(start_epoch, end_epoch):
        print('\nEpoch: %d' % epoch)

        net.train()
        train_loss = 0
        correct = 0
        total = 0

        
        batch_idx = 0
        inputs = None
        targets = None

        if epoch > prune_start :
            flag = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #print(count, epoch)
            if (count < prune_count) and (flag == 0) :
                #print(epoch, count, flag)
                outputs = net(inputs, True)
                count += 1
                flag = 1
            else:
                outputs = net(inputs, False)


            #loss = torch.mean(torch.max(0., 1. - outputs.data.numpy()*targets.data.numpy())**2)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        train_acc = 100.*correct/total

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = net(inputs, False)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        test_acc = 100.*correct/total 
        print("Epoch, Training accuracy, Test Accuracy", epoch, train_acc, test_acc)
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc
    print (net.conv1.mask)
    print (net.conv2.mask)

import torch
import torchvision
import torchvision.transforms as transforms
#from torch.cuda import device
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
#import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5,),(90.5,0.5,0.5))]
)

batch_size = 4
train_data = CIFAR10(root='/dataset', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = CIFAR10(root='/dataset', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('a')
#Define A Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,10,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,32,5)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 80)
        self.fc3 = nn.Linear(80, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) #flatten all dimesnsions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
cnn = CNN().to(device)
print('b')
#Loss Function And Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

#Training

for epoch in range(4):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #get the inputs; data is a list of [inputs, labels]
        inputs, labels =data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        #zero the parameter arguments (!!!Initialization ig !!!)
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print stats
        running_loss += loss.item()
        if i%2000 == 1999: #print every 2000 mini-batches
            print(f'[{epoch + 1}, {i+ 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss=0.0 #HUH???

print("Finished Training")

#Saving Model
#PATH = './cifar_net.pth' (!!!idk abt this!!)
#torch.save(cn.state_dict(), PATH)


#Testing on test data

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device=device)
        labels = labels.to(device=device)
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100* correct // total}%')
#Testing on test data by classes

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels =data;
        images = images.to(device=device)
        labels = labels.to(device=device)

        outputs = cnn(images)
        _, predictions = torch.max(outputs, 1)
        #total += labels.size(0)
        #correct == (predicted == labels).sum().item()
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

#print(f'Accuracy of the network on the 10000 test images: {100*correct // total}%')

for classname, correct_count in correct_pred.items():
    accuracy = 100*float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')



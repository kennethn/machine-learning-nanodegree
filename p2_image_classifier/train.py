'''
@Author: Ken Norton

Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class
probability.

Train a new network on a data set with train.py

Basic usage: 
    python train.py data_directory
    
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:

Set directory to save checkpoints: 
    python train.py data_dir --save_dir save_directory
    
Choose architecture: 
    python train.py data_dir --arch "vgg13"
    
Set hyperparameters: 
    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    
Use GPU for training: 
    python train.py data_dir --gpu
'''

import time
import json
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sns

from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(
    description='Train a new network on a data set')

# TODO: CL arguments

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

class_names = train_data.classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print('Loading pretrained model...................................')
    
# TODO: Build and train your network
model = models.vgg16(pretrained=True)
model.name = "vgg16"

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(0.5)),
            ('inputs', nn.Linear(25088, 120)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(120, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90, 80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

model.classifier = classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Using ', device, '...........................................')

model.to(device);

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 10
print_every = 5
steps = 0
running_loss = 0

print('Training model..............................................')

for e in range(epochs):
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            loss_v = 0
            accuracy = 0          
            
            for ii, (inputs_v, labels_v) in enumerate(validloader):
                optimizer.zero_grad()
                
                inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)
                model.to(device)
                with torch.no_grad():    
                    outputs = model.forward(inputs_v)
                    loss_v = criterion(outputs,labels_v)
                    ps = torch.exp(outputs).data
                    equals = (labels_v.data == ps.max(1)[1])
                    accuracy += equals.type_as(torch.FloatTensor()).mean()
                    
            loss_v = loss_v / len(validloader)
            accuracy = accuracy /len(validloader)          
                        
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {loss_v:.3f}.. "
                  f"Accuracy: {accuracy:.3f}")
                        
            running_loss = 0
            

# TODO: Do validation on the test set
def test_network(testloader):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d%%' % (100 * correct / total))


print('Validating on test set..........................................')

test_network(testloader)
            
model_f = 'model_checkpoint2.pth'

# TODO: Save the checkpoint
model.class_to_idx = trainloader.dataset.class_to_idx
model.epochs = epochs
checkpoint = {'batch_size': trainloader.batch_size,
              'input_size': 25088,
              'output_size': 120,
              'state_dict': model.state_dict(),
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': model.epochs,
              'classifier': model.classifier}

print('Saving model checkpoint.........................................')
torch.save(checkpoint, model_f)
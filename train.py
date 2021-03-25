import torch
from torch import optim
from torch import nn
import numpy as np
import os
import json
import matplotlib.pyplot as pyplt
from PIL import Image
from collections import OrderedDict
from torchvision import datasets,transforms, models

def load_categories():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_testing_data(base_data_dir):
    print ('Defining testing transforms and loading data...')
    testing_dir =  os.path.join(base_data_dir, 'test')
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    



def load_training_data(base_data_dir):
    print ("Define training transforms and loading data...")
    training_dir =  os.path.join(base_data_dir, 'train')
    validation_dir =  os.path.join(base_data_dir, 'valid')
    training_transforms =  transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])







# Create image folder datasets
training_dataset = datasets.ImageFolder(training_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(validation_dir, transform=testing_transforms)
testing_dataset =  datasets.ImageFolder(testing_dir, transform=testing_transforms)

#Define the dataloaders
training_dataloader =  torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)
validation_dataloader =  torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
testing_dataloader =  torch.utils.data.DataLoader(testing_dataset, batch_size=32, shuffle=True)



print ("Setup vgg_bn 16  model")
model = models.vgg16_bn(pretrained=True)
nn_input = model.classifier[0].in_features

print("Replace the network classifier")
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(nn_input, 256)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout(0.5)),
    ('fc2', nn.Linear(256, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

for param in model.parameters():
        param.requires_grad = False

model.classifier = classifier

print('Setup parameters')
learning_rate=0.001
epochs=2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 0
running_loss = 0
validation_loss=0
print_every = 50

print("Setup criterion and optimizer ")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


print('Model is Training...')

model.to(device);
if (device=="cuda"):
    tensor_type=torch.cuda.FloatTensor
else:
    tensor_type=torch.FloatTensor

for epoch in range(epochs):
    for inputs, labels in training_dataloader:
        steps += 1
        
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs_val, labels_val in validation_dataloader:

                    if torch.cuda.is_available():
                        inputs_val, labels_val = inputs_val.cuda(), labels_val.cuda()

                    logps = model.forward(inputs_val)
                    loss = criterion(logps, labels_val)
                    test_loss += loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels_val.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(tensor_type)).item()
                    
                    current_epoch = epoch+1
                    step_test_loss = test_loss/len(validation_dataloader)
                    step_accuracy = accuracy/len(validation_dataloader)
                    
            current_epoch = epoch+1
            training_loss = running_loss/print_every
            step_test_loss = test_loss/len(validation_dataloader)
            step_accuracy = accuracy/len(validation_dataloader)
           
            print(f'Epochs: {current_epoch}/{epochs}  .... Training Loss: {training_loss:.3f}',
                  f' .... Validation loss: {step_test_loss:.3f} .... Accuracy loss: {step_accuracy:.3f}')
            running_loss = 0
            model.train()
            

print(" Model Training accomplished ")

correct = 0
total = 0
model.to(device)
model.eval()
with torch.no_grad():
    for test_inputs, test_labels in testing_dataloader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
test_ds_accuracy = 100 * correct / total
print(f'Testing dataset accuracy {test_ds_accuracy:.2f}%')
model.train()


print("Saving checkpoint")
model.class_to_idx = training_dataset.class_to_idx

checkpoint = {
              'output_size': 102,
              'model_type': 'vgg16_bn',
              'classifier' : classifier,
              'learning_rate': learning_rate,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
             }

torch.save(checkpoint, 'checkpoint.pth')
print ("checkpoint saved")



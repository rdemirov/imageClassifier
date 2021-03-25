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

def load_checkpoint(checkpoint_path):
    print ("Loading model checkpoint from : {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['model_type'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = dict([(model.class_to_idx[clss], clss) for clss in model.class_to_idx])
    
    return model

model = load_checkpoint('checkpoint.pth')  
print ("Model checkpoint loaded")
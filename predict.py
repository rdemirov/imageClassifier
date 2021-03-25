import torch
from torchvision import models
from helpers import load_categories


cat_name = load_categories()

def load_checkpoint(checkpoint_path):
    """
    Rebuild model from checkpoint
    """
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

def predict(img_data, model,device, topk):
    """
    Classify image
    """
    model.to(device)
        
    model.eval()
    
    inputs = img_data.unsqueeze(0)
    inputs = inputs.to(device)
    
    output = model(inputs)
    ps = torch.exp(output).data
    
    ps_top = ps.topk(topk)
    idx_class = model.idx_to_class
    probs = ps_top[0].tolist()[0]
    classes=[]
    for i in ps_top[1].tolist()[0]:
        classes.append(idx_class[i])
    return probs, classes
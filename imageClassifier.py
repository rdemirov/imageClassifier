from argparse import ArgumentParser
import torch
import matplotlib.pyplot as pyplt
from PIL import Image
from helpers import process_image, load_categories
from predict import predict, load_checkpoint
from train import load_datasets, setup_model, train_model, test_network_accuracy, save_checkpoint



"""
Initialize command line arguments
"""
cmd_line_parser = ArgumentParser(description='Image Classifier arguments')


cmd_line_parser.add_argument('-v','--version' , action='version', help='Display application version', version='1.00')
cmd_line_parser.add_argument('-t','--train' , action='store_true', help='Setup and train the model', dest='train_mode')
cmd_line_parser.add_argument('-c','--classify' , action='store_false', help='Use the model to make a classification', dest='train_mode')
cmd_line_parser.add_argument('-data_dir','--base_data_dir' , default='flowers', help='Base data dir')
cmd_line_parser.add_argument('-g','--force_gpu' , action='store_true', help='Force GPU usage')

parsed_args = cmd_line_parser.parse_args()

cmd_line_parser.add_argument('-lr','--learning_rate' , default=0.001, help='NN Learning rate')
cmd_line_parser.add_argument('-e','--epochs' , default=2, help='Training epochs')
cmd_line_parser.add_argument('-hu','--hidden_units' , help='Hidden units')
cmd_line_parser.add_argument('-model','--network_model', choices=['vgg16_bn', 'densenet121', 'resnet101'], default='vgg16_bn', help='NN Model')
cmd_line_parser.add_argument('-cp_patch','--checkpoint', default='./checkpoint.pth', help='Checkpoint path')
cmd_line_parser.add_argument('-image','--image_path',  help='Image to classify' , required=(parsed_args.train_mode == False))
cmd_line_parser.add_argument('-topk','--topk',  help='Top K classes', default=5 )

parsed_args = cmd_line_parser.parse_args()

force_gpu = parsed_args.force_gpu
if(force_gpu == True):
        device="cuda"
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_mode = parsed_args.train_mode
topk = parsed_args.topk

if (train_mode == True):
    """
    Training mode
    """
    print ("Classifier Started in training mode")
    base_data_dir = parsed_args.base_data_dir
    learning_rate = parsed_args.learning_rate
    epochs = parsed_args.epochs
    hidden_units = parsed_args.hidden_units
    network_model = parsed_args.network_model
    # Load data 
    validation_dictionary, training_dictionary, testing_dictionary = load_datasets(base_data_dir)
    #Setup model
    model, criterion, optimizer =  setup_model(network_model, hidden_units, learning_rate)
    #Train model
    train_model(model, device,epochs, optimizer, criterion, training_dictionary, validation_dictionary)
    #Test network accuracy
    test_network_accuracy(model, device, testing_dictionary)
    #Save checkpoint
    save_checkpoint(network_model, model, training_dictionary,learning_rate,epochs, optimizer)


else:
    """
    classification mode
    """
    print ("Classifier Started in classification mode")
    print("Loading categories...")
    cat_to_name = load_categories()
    print("Loading model from checkpoint...")
    checkpoint_path = parsed_args.checkpoint
    model = load_checkpoint(checkpoint_path)
    image_path = parsed_args.image_path
    image_data = process_image(image_path)
    print ("Classifying image...")
    probabilities, classes = predict(image_data, model, device, topk)
    flower_names = [cat_to_name[category] for category in classes]
    y_axis = flower_names
    x_axis = probabilities
    fig,axes = pyplt.subplots(2,1)
    axes[0].set_title(flower_names[0])
    img_to_display = Image.open(image_path);
    axes[0].imshow(img_to_display)
    axes[1].barh(y_axis,x_axis)
    pyplt.show()
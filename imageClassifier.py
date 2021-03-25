from argparse import ArgumentParser
import sys



"""
Initialize command line arguments
"""
cmd_line_parser = ArgumentParser(description='Image Classifier arguments')


cmd_line_parser.add_argument('-v','--version' , action='version', help='Display application version', version='1.00')
cmd_line_parser.add_argument('-t','--train' , action='store_true', help='Setup and train the model', dest='train_mode')
cmd_line_parser.add_argument('-p','--predict' , action='store_false', help='Use the model to make a prediction', dest='train_mode')
cmd_line_parser.add_argument('-data_dir','--base_data_dir' , default='flowers', help='Base data dir')
cmd_line_parser.add_argument('-g','--force_gpu' , action='store_true', help='Force GPU usage')

parsed_args = cmd_line_parser.parse_args()

cmd_line_parser.add_argument('-lr','--learning_rate' , default=0.001, help='NN Learning rate')
cmd_line_parser.add_argument('-e','--epochs' , default=2, help='Training epochs')
cmd_line_parser.add_argument('-hu','--hidden-units' , help='Hidden units')
cmd_line_parser.add_argument('-model','--network_model', choices=['vgg16_bn', 'densenet121', 'resnet101'], default='vgg16_bn', help='NN Model')
cmd_line_parser.add_argument('-cp_patch','--checkpoint', default='./checkpoint.pth', help='Checkpoint path')
cmd_line_parser.add_argument('-image','--image_path',  help='Image to classify' , required=(parsed_args.train_mode == False))

parsed_args = cmd_line_parser.parse_args()


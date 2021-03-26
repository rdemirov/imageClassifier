# AI Programming with Python Project

This is the final project for the Udacity Python programming for AI nanodegree.
The project implements an image classifier for recognizing flower photos.The training,validation and testing datasets consist
of 102 flower categories. 

The project consists of 2 parts: 

## Jupyter notebook 

the notebook can be found in the project folder.It is named "Image Classifier Project".ipynb
To run it, go to the project folder and type:

```

jupyter notebook

```

In the web interface, locate the ipynb file and open it.


## Command line application 

The command line application entry point is imageClassifier.py .
It works in 2 modes - training and classification

### Training mode 

Classifier training mode is initiated by passing the '-t' parameter to the application:

```
python imageClassifier.py -t

```

The training mode of the classifier uses the following arguments: 

* --base_data_dir (-data_dir) Application directory housing the datasets. Must have subfolders named test,valid and train for each of the datasets.Default value : flowers
* --force_gpu ( -g )  Force the application to use gpu
* --learning_rate ( -lr )  Model learning rate. Default 0.001
* --epochs ( -e ) Training epochs. 2 by default 
* --hidden_units ( -hu) Hidden units 
* --network_model ( -model) Network architecture. vgg16_bn by default. Has 3 possible values: 'vgg16_bn', 'densenet121', 'resnet101'


### Classification mode 

Classification mode is used to make a prediction about the image. 
To start the classifier in classification mode, you should pass the '-c' arguments to the app:

```
python imageClassifier.py -c --image_path "./flowers/test/1/image_06743.jpg" 

```

Arguments accepted by the application for managing the classification process are:

* --checkpoint ( -cp_path ) path to the .pth file containing the model settings. By default it is ./checkpoint.pth
* --image_path (-image ) path to the image for which we intend to make a prediction
* --topk (-topk ) top K classes returned in the results , 5 by default
* --force_gpu ( -g )  Force the application to use gpu

#### Usage instructions 

Before trying to make a prediction, you should train the network.
Image classification can be made after the completion of the training

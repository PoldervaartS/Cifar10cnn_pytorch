# Cifar-10 Convolutional Neural Network

Neural Networks are immensely useful in many tasks, with a lot of the focus in recent years being Convolutional Neural Networks in computer vision. This project allows for training of various sizes of models by adding configurations and slight tweeks in the `main.py` and `Configure.py` files.  Currently configured to utilize 2 models that in ensemble can predict with 87% accuracy on the Cifar-10 dataset. 

Training will set up the model from scratch and begin training until the number of epochs set in the `training_configs` in the `Configure.py` (default 200). There are 2 types of resnet models, the first one (`resnet_version = 1`) is the standard block that utilizes 2 3x3 layers. The second one is the bottleneck model (`resnet_version=2`), utilizing 1x1 downscaling into a single 3x3 layer, back up to original dimensions with another 1x1. Both models then have the residual that was passed in added to the output.  

Test will use the 2 models trained (referred to by their model_dir and epoch_num) to predict the classes in training set, then find the accuracy of the ensembled predictions.

Logging Predictions will similarly use the 2 trained models, but will simply log the predictions in a [N, 10] array to file: `../predictions/prediction.npy` 

## Requirements

- pytorch 1.6
- tdqm
- sklearn

## How To Run
All commands must be executed inside of the `\code\` directory

### Training

 `python main.py train <dataset directory here>`

 Configutations are found in the `Configure.py` file for how deep you want the resnet to be, saving iterations, 
### Testing

`python main.py test <test data directory>`

models must be in the folder `../saved_models/deeper_model/` and `../saved_models/first_60_epoch/` for the First model and second model respectively. This ligns up with the values set in the `Configure.py` file as to how they should be set up.

### Logging Predictions 
Defaults saves to `../predictions/predictions.npy`

`python main.py predict <test data FILE PATH>`

### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs, model2_configs
import torch


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict", default='train')
parser.add_argument(
    "data_dir",
    help="path to the data",
    default=
    'C:\\Users\\Shane\\Documents\\PythonScripts\\Deeplearning\\Datasets\\CIFAR-10\\cifar-10-python\\cifar-10-batches-py'
)
parser.add_argument("--save_dir", help="path to save the results")
parser.add_argument("--result_dir", default="../predictions/", help="Path to save predictions")
args = parser.parse_args()

if __name__ == '__main__':
    print(torch.cuda.is_available())
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'

    model = MyModel(model_configs)
    model2 = MyModel(model2_configs)

    if args.mode == 'train':
        x_train, y_train, x_test, y_test = load_data(args.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(
            x_train, y_train)
        model.train(x_train, y_train, training_configs, x_valid, y_valid)
        model2.train(x_train, y_train, training_configs, x_valid, y_valid)

    elif args.mode == 'test':
        # Testing on public testing dataset
        _, _, x_test, y_test = load_data(args.data_dir)
        print("----- Predicting model 1 -----")
        model1_preds = model.predict_prob(x_test)
        print("----- Predicting model 2 -----")
        model2_preds = model2.predict_prob(x_test)

        # average the predictions of the models
        preds =  (model1_preds + model2_preds)/2
        preds = np.argmax(preds, axis=1)
        y = torch.tensor(y_test)
        torchpreds = torch.tensor(preds)
        print('Test accuracy: {:.4f}'.format(
            torch.sum(torchpreds == y) / y.shape[0]))

    elif args.mode == 'predict':
        # Predicting and storing results on private testing dataset
        x_test = load_testing_images(args.data_dir)
        predictions1 = model.predict_prob(x_test, mode="test")
        predictions2 = model2.predict_prob(x_test, mode="test")
        predictions = (predictions1 + predictions2) / 2
        os.makedirs(args.result_dir, exist_ok=True)
        np.save(args.result_dir + "predictions", predictions)

### END CODE HERE

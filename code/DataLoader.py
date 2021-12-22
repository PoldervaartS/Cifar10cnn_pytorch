import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from ImageUtils import parse_record, preprocess_image
"""This script implements the functions for reading data.
"""


def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    print('----Loading Data----')
    x_train = np.empty((0, 3072))
    y_train = np.empty(0)
    for i in range(1, 6):
        with open(data_dir + "\\data_batch_" + str(i), 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            x_train = np.append(x_train, data[b'data'], axis=0)
            y_train = np.append(y_train, data[b'labels'], axis=0)

    x_test = None
    y_test = None
    with open(data_dir + "\\test_batch", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        x_test = np.array(data[b'data'], dtype=np.float32)
        y_test = np.array(data[b'labels'], dtype=np.int32)

    print(f'X shape: {x_test.shape}')
    print(f'Y shape: {y_test.shape}')
    return (x_train, y_train, x_test, y_test)
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = np.load(os.path.join(data_dir))
    x_test = np.array([ x.reshape(32,32,3) for x in x_test], dtype=np.float32)
    x_test -= np.mean(x_test)
    x_test /= np.std(x_test)
    # np is already in 32x32
    return x_test
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    ### YOUR CODE HERE
    x_train_new, x_valid, y_train_new, y_valid = train_test_split(x_train,
                                                                  y_train,
                                                                  test_size=1 -
                                                                  train_ratio)
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

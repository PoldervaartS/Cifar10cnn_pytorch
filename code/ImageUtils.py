import time
import numpy as np
from numpy.core.shape_base import block
from torchvision.transforms import transforms
from torchvision import transforms
from RandAugment import RandAugment
from matplotlib import pyplot as plt
"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    ### END CODE HERE

    image = preprocess_image(image, training)  # If any.

    return image


def preprocess_image(image, training=False):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    # Probably going to be doing more here
    if training:

        image = np.pad(image, ((4, 4), (4, 4), (0, 0)),
                       'constant',
                       constant_values=(0))
        ### YOUR CODE HERE

        data_transforms = transforms.Compose([
        transforms.ToTensor(),
            RandAugment(3,5),
        transforms.ColorJitter(brightness=0.75, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomRotation(20),
        transforms.RandomCrop(32, padding=4),
        ])

        # image = np.transpose(data_transforms(image).numpy(), [1, 2, 0])

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        xindex = np.random.randint(0, image.shape[0] - 32)
        yindex = np.random.randint(0, image.shape[1] - 32)
        image = image[xindex:(xindex + 32), yindex:(yindex + 32), :]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image -= np.mean(image)
    image /= np.std(image)
    ### END CODE HERE

    # print(image)
    # plt.imshow(image)
    # plt.show()
    # time.sleep(5)

    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE
# import tensorflow as tf
from numpy.core.shape_base import block
import torch.nn as nn
from torch.functional import Tensor
"""This script defines the network.
"""


class MyNetwork(nn.Module):
    def __init__(self, configs):
        self.configs = configs

        super(MyNetwork, self).__init__()
        self.resnet_size = configs["resnet_size"]
        self.num_classes = configs["num_classes"]
        self.first_num_filters = configs["first_num_filters"]
        self.resnet_version = configs["resnet_version"]

        # YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(3,
                                     self.first_num_filters, (3, 3),
                                     padding=(1, 1)).cuda()
        # YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            block_fn = standard_block
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters,
                eps=1e-5,
                momentum=0.997,
            )
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(
                stack_layer(filters, block_fn, strides, self.resnet_size,
                            self.first_num_filters))
        self.output_layer = output_layer(filters * 4, self.resnet_version, self.num_classes)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs


class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        # YOUR CODE HERE
        self.subnet = nn.Sequential(
            nn.BatchNorm2d(num_features, eps, momentum), nn.ReLU()).cuda()
        # YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        # YOUR CODE HERE
        return self.subnet(inputs)
        # YOUR CODE HERE


class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides,
                 first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        if projection_shortcut is not None:
            self.projection_shortcut = projection_shortcut
        else:
            self.projection_shortcut = nn.Identity().cuda()
            strides = 1
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        self.subnet = nn.Sequential(
            batch_norm_relu_layer(first_num_filters).cuda(),
            nn.Conv2d(first_num_filters, filters, (3, 3), strides,
                      (1, 1)).cuda(),
            nn.Dropout2d(p=0.5),
            batch_norm_relu_layer(filters).cuda(),
            nn.Conv2d(filters, filters, (3, 3), (1, 1), (1, 1)).cuda(),
        )
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        return self.subnet(inputs).add(self.projection_shortcut(inputs))
        # output is the sampled stuff + the input

        ### YOUR CODE HERE


class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
                strides: A positive integer. The stride to use for the block. If
                        greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides,
                 first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        # YOUR CODE HERE
        if projection_shortcut is not None:
            self.projection_shortcut = projection_shortcut
            self.bnsmallinput = batch_norm_relu_layer(first_num_filters).cuda()
            self.isupscaled = True
        else:
            self.projection_shortcut = nn.Identity().cuda()
            self.isupscaled = False
            strides = 1

        # SUBNET IS NOT GETTING ITS WEIGHTS UPDATED FOR WHATEVER REASON AAAAHHHH
        self.subnet = nn.Sequential(
            batch_norm_relu_layer(filters).cuda(),
            nn.Conv2d(filters, first_num_filters, (1, 1), (1, 1)).cuda(),
            batch_norm_relu_layer(first_num_filters).cuda(),
            nn.Conv2d(first_num_filters, first_num_filters, (3, 3), (1, 1),
                      (1, 1)).cuda(),
            batch_norm_relu_layer(first_num_filters).cuda(),
            nn.Conv2d(first_num_filters, filters, (1, 1), (1, 1)).cuda())

        # self.bnrfilter = batch_norm_relu_layer(filters).cuda()
        # self.convobigsmall = nn.Conv2d(
        #     filters, first_num_filters, (1, 1), (1, 1)).cuda()
        # self.bnrconvobigsmall = batch_norm_relu_layer(first_num_filters).cuda()
        # self.convosmallsmall = nn.Conv2d(
        #     first_num_filters, first_num_filters, (3, 3), (1, 1), (1, 1)).cuda()
        # self.bnrconvosmallsmall = batch_norm_relu_layer(
        #     first_num_filters).cuda()
        # self.convosmallbig = nn.Conv2d(
        #     first_num_filters, filters, (1, 1), (1, 1)).cuda()
        # self.bnrconvobig = batch_norm_relu_layer(filters).cuda()
        # have the projections

        # YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        # YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if self.isupscaled:
            inputs = self.bnsmallinput(inputs)
            inputs = self.projection_shortcut(inputs)
            return self.subnet(inputs).add(inputs)
        else:
            return self.subnet(inputs).add(inputs)


        # only doing subnet(inputs) was doing better earlier
        # File "C:\Users\Shane\Dropbox\School\2021-2022Year\Fall\DeepLearning\Project\starter_code\Model.py", line 111, in save
#     checkpoint_path = os.path.join(self.config.modeldir,
# AttributeError: 'MyModel' object has no attribute 'config'

# YOUR CODE HERE



# POTENTIAL TODO to test a different resnet implementation
class wide_block(nn.Module):
    def __init__(self, filters, projection_shortcut, strides,
                 first_num_filters) -> None:
        super(wide_block, self).__init__()

        self.wideblock = nn.Sequential(
            nn.Conv2d()
        )



    def forward(self, inputs: Tensor) -> Tensor:


        return self.wideblock(inputs).add(inputs)


class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
                            convolution in a block.
                block_fn: 'standard_block' or 'bottleneck_block'.
                strides: A positive integer. The stride to use for the first block. If
                                greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size,
                 first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        # END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        filters_in = filters if filters == first_num_filters else filters_out // 2
        print(f'filters in: {filters_in}\tfilters out: {filters_out}')
        self.resnet_size = resnet_size

        projection_shortcut = nn.Conv2d(
            filters_in, filters_out, (1, 1),
            1 if filters_in == filters_out else strides).cuda()

        if block_fn is standard_block:
            self.layers = nn.ModuleList([
                block_fn(filters_out, projection_shortcut if i == 0 else None,
                         strides, filters_in if i == 0 else filters_out)
                for i in range(resnet_size)
            ])
        else:
            self.layers = nn.ModuleList([
                block_fn(filters_out, projection_shortcut if i == 0 else None,
                        strides, filters_in) for i in range(resnet_size)
            ])
        # END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        # END CODE HERE
        output = inputs
        for i in range(self.resnet_size):
            output = self.layers[i](output)

        return output
        # END CODE HERE


class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters,
                                                 eps=1e-5,
                                                 momentum=0.997)
            num_filters = filters
        else:
            num_filters = filters // 4

        # avgpool flatten then linear
        # self.avgPool = nn.AvgPool2d(8)
        # self.flatten = nn.Flatten()
        # self.outputLayer = nn.Linear(num_filters, num_classes)
        self.subnet = nn.Sequential(nn.AvgPool2d(8), nn.Flatten(),
                                    nn.Linear(num_filters, num_classes)).cuda()
        # END CODE HERE

        # END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        # END CODE HERE
        output = inputs if not hasattr(self,
                                       'bn_relu') else self.bn_relu(inputs)

        return self.subnet(output)
        # END CODE HERE


### END CODE HERE
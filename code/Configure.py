# Below configures are examples,
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
    "name": 'MyModel',
    "save_dir": '../saved_models/deeper_model2/',
    "resnet_size": 8,
    "num_classes": 10,
    "first_num_filters": 16,
    "save_interval": 15,
    "weight_decay": 2E-4,
    "model_dir": '../saved_models/deeper_model/',
    "model": 'bottleneck',
    "epoch_num": 195,
    "resnet_version": 2
    # ...
}

training_configs = {
    "batch_size": 200
    # ...
}

model2_configs = {
    "name": 'MyModel',
    "save_dir": '../saved_models/first_60_epoch',
    "resnet_size": 5,
    "num_classes": 10,
    "first_num_filters": 16,
    "save_interval": 5,
    "weight_decay": 2E-4,
    "model_dir": '../saved_models/first_60_epoch/',
    "model": 'bottleneck',
    "epoch_num": 175,
    "resnet_version": 2
}
### END CODE HERE
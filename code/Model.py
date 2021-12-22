### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import tqdm
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from matplotlib import pyplot as plt
"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)

        self.loss = nn.CrossEntropyLoss()
        # TODO pass in learning rate
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=0.01,
                                          weight_decay=configs["weight_decay"])

    # make it so that I can load in the stuff.
    def model_setup(self):
        checkpointfile = os.path.join(
            self.configs["model_dir"],
            'model-%d.ckpt' % (self.configs["epoch_num"]))
        self.load(checkpointfile)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        num_samples = x_train.shape[0]
        num_batches = num_samples // configs["batch_size"]
        max_epoch = 200


        print('### Training... ###')
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]


            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch % 80 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] /= 10
            ### YOUR CODE HERE

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                x_batch = np.array([
                    parse_record(x, True)
                    for x in curr_x_train[i * configs["batch_size"]:(i + 1) *
                                          configs["batch_size"]]
                ])
                y_batch = curr_y_train[i * configs["batch_size"]:(i + 1) *
                                       configs["batch_size"]]
                x_batch = np.transpose(x_batch, ([0, 3, 1, 2]))

                # print(x_batch.shape)
                # print(y_batch[0])
                # plt.imshow(np.transpose(x_batch[0], [1,2,0] ))
                # plt.show()
                # time.sleep(5)

                # print(self.network.stack_layers[0].layers[0].subnet[1].weight)
                outputs = self.network(torch.tensor(x_batch).float().cuda())
                loss = self.loss(outputs, torch.tensor(y_batch).long().cuda())

                ### YOUR CODE HERE

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(self.network.stack_layers[0].layers[0].subnet[1].weight)
                # print(self.network)

                print('Batch {:d}/{:d} Loss {:.6f}'.format(
                    i, num_batches, loss),
                      end='\r',
                      flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
                epoch, loss, duration))

            if epoch % self.configs["save_interval"] == 0:
                self.save(epoch)
        pass

    def evaluate(self, x, y):
        self.network.eval()
        self.optimizer.zero_grad(set_to_none=True)
        print('### Test or Validation ###')
        checkpointfile = os.path.join(self.configs["model_dir"],
                                      'model-%d.ckpt' % (self.configs["epoch_num"]))
        self.load(checkpointfile)

        preds = np.empty(x.shape[0], dtype=np.int32)
        x = np.array([parse_record(xval, False) for xval in x])
        x = np.transpose(x, ([0, 3, 1, 2]))
        with torch.no_grad():
            # set range to be 10 in order to figure out the preds issue
            for i in tqdm.tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                preds[i] = self.network(
                    torch.tensor(
                        x[i]).unsqueeze(0).float().cuda()).argmax().cpu().numpy()
            ### END CODE HERE

        y = torch.tensor(y)
        torchpreds = torch.tensor(preds)
        print('Test accuracy: {:.4f}'.format(
            torch.sum(torchpreds == y) / y.shape[0]))

    def predict_prob(self, x, mode = "train"):
        self.network.eval()
        self.optimizer.zero_grad(set_to_none=True)
        print('### Test or Validation ###')
        checkpointfile = os.path.join(
            self.configs["model_dir"],
            'model-%d.ckpt' % (self.configs["epoch_num"]))
        self.load(checkpointfile)

        # returns n x 10 which is the
        if mode=="test":
            x= np.transpose(x, ([0, 3, 1, 2]))
        else:
            x = np.transpose(np.array([parse_record(xval, False) for xval in x]), ([0, 3, 1, 2]))
        predictions = np.empty( (x.shape[0], 10), dtype=np.float32)
        with torch.no_grad():

            for i in tqdm.tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                predictions[i] = self.network(
                    torch.tensor(x[i]).unsqueeze(
                        0).float().cuda()).cpu().numpy()
        return predictions

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs['model_dir'],
                                       'model-%d.ckpt' % (epoch))
        os.makedirs(self.configs['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE
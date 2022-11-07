import pickle, gzip, numpy as np

import util
from Network import Network

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

if __name__ == "__main__":
    np.seterr(all='raise')
    network = Network([784, 100, 10], 0.005, 10)
    epoch_map = network.train(train_set[0], train_set[1], 32, valid_set[0], valid_set[1])
    best_epoch = max(epoch_map.values(), key=lambda x: x[1][0])
    for idx, epoch in enumerate(epoch_map.values()):
        print("Epoch ", idx, ":", epoch[1])
    best_epoch_index = [i for i in epoch_map if epoch_map[i] == best_epoch][0]
    print("Best epoch results on validation set: epoch", best_epoch_index, ' with result ', best_epoch[1])
    print("Test on test data ",
          best_epoch[0].test(test_set[0], test_set[1]))

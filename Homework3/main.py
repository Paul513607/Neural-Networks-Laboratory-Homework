import pickle, gzip, numpy as np

import util
from Network import Network

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


def augment_train_set(train_set):
    train_set_augmented = [item for item in train_set[0]]
    train_labels_augmented = [item for item in train_set[1]]

    counter = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for image, label in zip(train_set[0], train_set[1]):
            counter += 1
            shifted_image = util.shift_image(image, dx, dy)
            train_set_augmented.append(shifted_image)
            train_labels_augmented.append(label)

    # for angle in [-15, 15, -30, 30, -45, 45]:
    #     for image, label in zip(train_set[0], train_set[1]):
    #         counter += 1
    #         rotated_image = util.rotate_image(image, angle)
    #         train_set_augmented.append(rotated_image)
    #         train_labels_augmented.append(label)

    return np.array(train_set_augmented), np.array(train_labels_augmented)


if __name__ == "__main__":
    # Recommend 50 epochs
    network = Network([784, 100, 10], 0.005, 50)
    train_set_data, train_set_labels = augment_train_set(train_set)
    print("Augmentation done. Augmented train set size: ", len(train_set_data))
    epoch_map = network.train(train_set_data, train_set_labels, 32, valid_set[0], valid_set[1])
    best_epoch = max(epoch_map.values(), key=lambda x: x[1][0])
    for idx, epoch in enumerate(epoch_map.values()):
        print("Epoch ", idx, ":", epoch[1])
    best_epoch_index = [i for i in epoch_map if epoch_map[i] == best_epoch][0]
    print("Best epoch results on validation set: epoch", best_epoch_index, ' with result ', best_epoch[1])
    print("Test on test data ",
          best_epoch[0].test(test_set[0], test_set[1]))

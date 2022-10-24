import pickle, gzip, numpy as np

from perceptron import Perceptron

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


if __name__ == '__main__':
    perceptron_list = []
    for i in range(0, 10):
        perceptron = Perceptron(i)
        perceptron.mini_batch(train_set[0], train_set[1])
        print("[Train] Accuracy for ", i, ": ", perceptron.accuracy(train_set[0], train_set[1]))
        perceptron_list.append(perceptron)

    for i in range(0, 10):
        print("[Validate] Accuracy for ", i, ": ", perceptron_list[i].accuracy(valid_set[0], valid_set[1]))

    for i in range(0, 10):
        print("[Test] Accuracy for ", i, ": ", perceptron_list[i].accuracy(test_set[0], test_set[1]))





import math

import numpy as np


class Neuron:
    class NeuronException(Exception):
        pass

    def __init__(self, number_of_inputs, weights=None, biase=None, biase_weight=None, training_rate=.5):
        if weights and len(weights) != number_of_inputs:
            raise self.NeuronException('number of inputs and length of supplied weights must be the same')
        self.weights = weights if weights is not None else list(np.random.normal(0, .5, number_of_inputs))
        self.biase = biase
        self.biase_weight = biase_weight if biase_weight is not None else np.random.normal(0, .5)
        self.training_rate = training_rate

    def _activation(self, x):
        return 1 / (1 + math.exp(-x))
        # return 0 if x < 0 else 1

    def guess(self, inputs):
        if len(inputs) != len(self.weights):
            raise self.NeuronException('input array must be the same length as weights array')
        total = sum([inputs[i] * self.weights[i] for i in range(len(self.weights))])
        if self.biase:
            total += self.biase * self.biase_weight
        return self._activation(total)

    def predict(self, inputs):
        return round(self.guess(inputs))

    def adjust_weights(self, data, error, training_rate=None):
        training_rate = training_rate if training_rate is not None else self.training_rate
        for i in range(len(self.weights)):
            self.weights[i] += (data[i] * error) * training_rate
        if self.biase:
            self.biase_weight += self.biase * error * training_rate

    def train(self, x_train, y_train, training_cycles=2):
        for cycle_number in range(training_cycles):
            local_training_rate = self.training_rate / (cycle_number + 1)
            for d, data in enumerate(x_train):
                prediction = self.guess(data)
                error = y_train[d] - prediction
                if error != 0:
                    self.adjust_weights(data, error, local_training_rate)

    def test(self, x_test, y_test):
        error_points = []
        for i, data in enumerate(x_test):
            if self.predict(data) != y_test[i]:
                error_points.append(data)
        return len(error_points) / len(x_test), error_points

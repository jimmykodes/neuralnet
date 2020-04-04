import math
import random


class BaseNeuron:
    class NeuronException(Exception):
        pass

    def __init__(self, number_of_inputs, training_rate=.1, weights=None):
        if weights and len(weights) != number_of_inputs:
            raise self.NeuronException('number of inputs and length of supplied weights must be the same')

        self.number_of_inputs = number_of_inputs
        self.weights = weights.copy() if weights is not None else [random.normalvariate(0, .5) for _ in range(self.number_of_inputs)]
        self.training_rate = training_rate
        self.weight_history = [self.weights.copy()]

    def _activation(self, x):
        return 1 / (1 + math.exp(-x))

    def guess(self, inputs):
        if len(inputs) != self.number_of_inputs:
            raise self.NeuronException('input array must be the same length as weights array')
        total = sum([inputs[i] * self.weights[i] for i in range(len(self.weights))])
        return self._activation(total)

    def predict(self, inputs):
        return round(self.guess(inputs))

    def adjust_weights(self, data, error):
        for i in range(len(self.weights)):
            self.weights[i] += (data[i] * error) * self.training_rate
        self.weight_history.append([weight for weight in self.weights])


class Perceptron(BaseNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_error = []

    def train(self, x_train, y_train, training_cycles=2):
        for _ in range(training_cycles):
            for i, data in enumerate(x_train):
                prediction = self.guess(data)
                error = y_train[i] - prediction
                self.training_error.append(error)
                if error != 0:
                    self.adjust_weights(data, error)

    def test(self, x_test, y_test):
        error_points = []
        for i, data in enumerate(x_test):
            if self.predict(data) != y_test[i]:
                error_points.append(data)
        return len(error_points) / len(x_test), error_points

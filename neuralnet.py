import math
import random

from matrix import Matrix


class NeuralNet:
    def __init__(self, num_inputs, num_hidden, num_output):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_weights = Matrix(self.num_hidden, self.num_inputs)
        self.output_weights = Matrix(self.num_output, self.num_hidden)
        self.hidden_weights.map(lambda x: random.normalvariate(0, .5))
        self.output_weights.map(lambda x: random.normalvariate(0, .5))

    def _activation(self, x):
        return 1 / (1 + math.exp(-x))

    def guess(self, inputs):
        hidden = self.hidden_weights * inputs
        hidden = list(map(lambda x: self._activation(x), hidden))
        output = self.output_weights * hidden
        output = list(map(lambda x: self._activation(x), output))
        return output

    def predict(self, inputs):
        guess = self.guess(inputs)
        return list(map(lambda x: round(x), guess))

    def train(self, inputs, targets, training_rate=.1):
        hidden = self.hidden_weights * inputs
        guess = self.output_weights * hidden

        hidden = list(map(lambda x: self._activation(x), hidden))
        guess = list(map(lambda x: self._activation(x), guess))

        h = Matrix.from_array([hidden]).transpose()
        g = Matrix.from_array([guess]).transpose()
        t = Matrix.from_array([targets]).transpose()

        t_weights = self.output_weights.transpose()
        output_error = Matrix.from_array(g - t)
        hidden_error = Matrix.from_array(t_weights * output_error)

        output_gradient = Matrix.from_array(output_error.matrix)
        hidden_gradient = Matrix.from_array(hidden_error.matrix)

        output_gradient.map(lambda x: x * (1 - x))
        hidden_gradient.map(lambda x: x * (1 - x))

        delta_output_weights = output_gradient * h.transpose()
        delta_output_weights = Matrix.from_array(delta_output_weights)
        delta_output_weights.map(lambda x: x * training_rate)

        delta_hidden_weights = hidden_gradient * Matrix.from_array([inputs])
        delta_hidden_weights = Matrix.from_array(delta_hidden_weights)
        delta_hidden_weights.map(lambda x: x * training_rate)

        self.output_weights.matrix = self.output_weights + delta_output_weights
        self.hidden_weights.matrix = self.hidden_weights + delta_hidden_weights

    def test(self, inputs, targets):
        print(self.predict(inputs))
        if self.predict(inputs) == targets:
            return 0
        else:
            return 1

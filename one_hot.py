from neuron import Neuron
import numpy as np


class OneHotNeuralNet:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        weights = list(np.random.normal(0, .5, size=self.number_of_inputs))
        self.neurons = [Neuron(number_of_inputs=number_of_inputs, weights=weights.copy(), location=i) for i in range(number_of_neurons)]

    def train(self, x_train, y_train, training_cycles=2):
        for cycle_number in range(training_cycles):
            for i, data in enumerate(x_train):
                desired = [0 for _ in range(self.number_of_neurons)]
                desired[y_train[i]] = 1
                for neuron in self.neurons:
                    neuron.step_train(data, desired, training_rate=.5 / (cycle_number + 1))
        return [neuron.training_error for neuron in self.neurons]

    def test(self, x_test, y_test):
        error_points = []
        for i, data in enumerate(x_test):
            desired = [0 for _ in range(self.number_of_neurons)]
            desired[y_test[i]] = 1
            output = []
            for neuron in self.neurons:
                output.append(neuron.predict(data))
            error = [desired[i] - output[i] for i in range(self.number_of_neurons)]
            if any(error):
                error_points.append(data)
        return len(error_points) / len(x_test), error_points

from neuron import BaseNeuron as Neuron


class OneHotNeuralNet:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.neurons = [Neuron(number_of_inputs=number_of_inputs) for _ in range(number_of_neurons)]
        self.error_output = []

    def train(self, x_train, y_train, training_cycles=2):
        for _ in range(training_cycles):
            for i, data in enumerate(x_train):
                desired = [0 for _ in range(self.number_of_neurons)]
                desired[y_train[i]] = 1
                guesses = [neuron.predict(data) for neuron in self.neurons]
                error = [desired[i] - guesses[i] for i in range(self.number_of_neurons)]
                for e, neuron in enumerate(self.neurons):
                    neuron.adjust_weights(data, error[e])

    def test(self, x_test, y_test):
        error_points = []
        for i, data in enumerate(x_test):
            desired = [0 for _ in range(self.number_of_neurons)]
            desired[y_test[i]] = 1
            output = [neuron.predict(data) for neuron in self.neurons]
            if desired != output:
                error_points.append(data)
                self.error_output.append([desired, output])
        return len(error_points) / len(x_test), error_points

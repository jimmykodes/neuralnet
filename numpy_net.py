import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Net:
    def __init__(self, num_inputs, num_hidden, num_output):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.w1 = np.random.rand(self.num_inputs, self.num_hidden)
        self.w2 = np.random.rand(self.num_hidden, self.num_output)
        self.a2 = None
        self.z2 = None
        self.z3 = None
        self.yHat = None

    def _activate(self, z):
        return 1 / (1 + np.exp(-z))

    def _d_activate(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def forward(self, X):
        self.z2 = np.dot(X, self.w1)
        self.a2 = self._activate(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        return self._activate(self.z3)

    def cost_function(self, X, y):
        self.yHat = self.forward(X)
        return 0.5 * sum((y - self.yHat) ** 2)

    def d_cost_function(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self._d_activate(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.w2.T) * self._d_activate(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2


iris = load_iris()
X = iris.data
y = np.reshape(iris.target, (150, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y)

n = Net(4, 5, 3)
cost = n.cost_function(X_test, y_test)
print(cost)
dJdW1, dJdW2 = n.d_cost_function(X_train, y_train)
n.w1 -= dJdW1
n.w2 -= dJdW2

cost = n.cost_function(X_test, y_test)
print(cost)

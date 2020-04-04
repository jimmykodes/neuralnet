from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from neuralnet import NeuralNet

n = NeuralNet(4, 3, 3)

iris = load_iris()
target = []
for t in iris.target:
    desired = [0, 0, 0]
    desired[t] = 1
    target.append(desired)
data = normalize(iris.data)
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=.7)

for i, inputs in enumerate(x_train):
    n.train(inputs.tolist(), y_train[i])
for i, inputs in enumerate(x_train):
    n.train(inputs.tolist(), y_train[i])
for i, inputs in enumerate(x_train):
    n.train(inputs.tolist(), y_train[i])

errors = 0
for i, inputs in enumerate(iris.data):
    errors += n.test(inputs.tolist(), target[i])

print(errors)

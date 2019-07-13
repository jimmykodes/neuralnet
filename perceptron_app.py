import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neuron import Neuron

iris = datasets.load_iris()
# perfect weights: [6.369765514857479, 2.9423096883734834, -7.311728871619155, -7.812221118696072] for 50 - 150
lower_bound = 0
upper_bound = 100
iris_x_data = iris.data[lower_bound:upper_bound]
iris_y_data = [1 if i == 1 else 0 for i in iris.target[lower_bound:upper_bound]]
x_train, x_test, y_train, y_test = train_test_split(iris_x_data, iris_y_data)

n = Neuron(number_of_inputs=4, weights=list(np.random.normal(0, .5, size=4)))
n.train(x_train, y_train, training_cycles=4)
error, error_points = n.test(iris_x_data, iris_y_data)
print(f'error: {error}')

if error < .5:
    plot_data = []
    for i, data in enumerate(iris_x_data):
        plot_data.append([*data, iris.target_names[iris_y_data[i]]])
    for point in error_points:
        plot_data.append([*point, 'error'])
    df = pd.DataFrame(plot_data, columns=[*iris.feature_names, 'species'])
    g = sns.pairplot(data=df, hue='species', palette={'error': 'red', 'setosa': 'green', 'versicolor': 'blue', 'virginica': 'purple'})
    g.fig.show()
else:
    print('error too high. not plotting')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from one_hot import OneHotNeuralNet


def plot_error(data):
    error_df1 = pd.DataFrame(data[0], columns=['n_1'])
    error_df2 = pd.DataFrame(data[1], columns=['n_2'])
    error_df3 = pd.DataFrame(data[2], columns=['n_3'])

    plt.subplot(3, 1, 1)
    sns.lineplot(data=error_df1)
    plt.subplot(3, 1, 2)
    sns.lineplot(data=error_df2)
    plt.subplot(3, 1, 3)
    sns.lineplot(data=error_df3)
    plt.show()


def plot_results(iris, error_points):
    plot_data = []
    for i, data in enumerate(iris.data):
        plot_data.append([*data, iris.target_names[iris.target[i]]])
    for point in error_points:
        plot_data.append([*point, 'error'])

    data_df = pd.DataFrame(plot_data, columns=[*iris.feature_names, 'species'])
    p_plot = sns.pairplot(
        data=data_df,
        hue='species',
        palette={'error': 'red', 'setosa': 'green', 'versicolor': 'blue', 'virginica': 'purple'}
    )
    p_plot.fig.show()


def plot_weights(network):
    plot_data1 = np.transpose([[hist[0] for hist in neuron.weight_history] for neuron in network.neurons])
    plot_data2 = np.transpose([[hist[1] for hist in neuron.weight_history] for neuron in network.neurons])
    plot_data3 = np.transpose([[hist[2] for hist in neuron.weight_history] for neuron in network.neurons])
    plot_data4 = np.transpose([[hist[3] for hist in neuron.weight_history] for neuron in network.neurons])
    df1 = pd.DataFrame(plot_data1, columns=['n1', 'n2', 'n3'])
    df2 = pd.DataFrame(plot_data2, columns=['n1', 'n2', 'n3'])
    df3 = pd.DataFrame(plot_data3, columns=['n1', 'n2', 'n3'])
    df4 = pd.DataFrame(plot_data4, columns=['n1', 'n2', 'n3'])
    plt.subplot(4, 1, 1)
    sns.lineplot(data=df1, dashes=False)
    plt.subplot(4, 1, 2)
    sns.lineplot(data=df2, legend=False, dashes=False)
    plt.subplot(4, 1, 3)
    sns.lineplot(data=df3, legend=False, dashes=False)
    plt.subplot(4, 1, 4)
    sns.lineplot(data=df4, legend=False, dashes=False)
    plt.show()


def main():
    iris = datasets.load_iris()
    data = iris.data
    x_train, x_test, y_train, y_test = train_test_split(data, iris.target)
    network = OneHotNeuralNet(number_of_inputs=4, number_of_neurons=3)
    network.train(x_train, y_train, training_cycles=10)
    error, error_points = network.test(data, iris.target)
    accuracy = (1 - error) * 100
    print(f'Accuracy: {accuracy:0.2f}%\nNumber of errors: {len(error_points)}')
    print(network.error_output)
    if accuracy > 70:
        weights = [neuron.weights for neuron in network.neurons]
        print(weights)
    # plot_weights(network)
    # if len(error_points) >= 50:
    #     print('too many errors, not plotting')
    #     return
    plot_results(iris, error_points)


if __name__ == '__main__':
    main()

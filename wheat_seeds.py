import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


# def plot_results(iris, error_points):
#     plot_data = []
#     for i, data in enumerate(iris.data):
#         plot_data.append([*data, iris.target_names[iris.target[i]]])
#     for point in error_points:
#         plot_data.append([*point, 'error'])
#
#     data_df = pd.DataFrame(plot_data, columns=[*iris.feature_names, 'species'])
#     p_plot = sns.pairplot(
#         data=data_df,
#         hue='species',
#         palette={'error': 'red', 'setosa': 'green', 'versicolor': 'blue', 'virginica': 'purple'}
#     )
#     p_plot.fig.show()


def main():
    with open('data/seed_data.json') as f:
        seeds = json.load(f)

    for point in seeds['data']:
        assert len(point) == 7, point
    x_train, x_test, y_train, y_test = train_test_split(seeds['data'], seeds['target'], train_size=0.7)
    network = OneHotNeuralNet(number_of_inputs=7, number_of_neurons=3)
    error_rate = network.train(x_train, y_train, training_cycles=5)
    error, error_points = network.test(x_test, y_test)
    print(f'Accuracy: {100 - (error * 100):0.1f}%\nNumber of errors: {len(error_points)}')
    plot_error(error_rate)
    # if len(error_points) >= 50:
    #     print('too many errors, not plotting')
    #     return
    # plot_results(iris, error_points)


if __name__ == '__main__':
    main()

import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from one_hot import OneHotNeuralNet


def main():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    network = OneHotNeuralNet(number_of_inputs=4, number_of_neurons=3)
    network.train(x_train, y_train)
    error, error_points = network.test(iris.data, iris.target)
    print(f'Error Percentage: {error * 100}\nNumber of errors: {len(error_points)}')
    if len(error_points) >= 50:
        print('too many errors, not plotting')
        return

    plot_data = []
    for i, data in enumerate(iris.data):
        plot_data.append([*data, iris.target_names[iris.target[i]]])
    for point in error_points:
        plot_data.append([*point, 'error'])

    df = pd.DataFrame(plot_data, columns=[*iris.feature_names, 'species'])
    g = sns.pairplot(
        data=df,
        hue='species',
        palette={'error': 'red', 'setosa': 'green', 'versicolor': 'blue', 'virginica': 'purple'}
    )
    g.fig.show()


if __name__ == '__main__':
    main()

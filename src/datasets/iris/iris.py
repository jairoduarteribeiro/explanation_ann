import pandas as pd

from datasets.dataset_utils import get_dataset_path, save_dataset, split_dataset, transform


def load_data():
    path = get_dataset_path('iris', 'iris.csv')
    column_names = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target')
    df = pd.read_csv(path, header=None, names=column_names)
    df.loc[df.target == 'Iris-setosa', 'target'] = 0
    df.loc[df.target == 'Iris-versicolor', 'target'] = 1
    df.loc[df.target == 'Iris-virginica', 'target'] = 2
    df['target'] = df['target'].astype('int')
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x = transform(x, x.columns)
    return split_dataset(x, y)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    train_csv_path = get_dataset_path('iris', 'train.csv')
    test_csv_path = get_dataset_path('iris', 'test.csv')
    save_dataset(x_train, y_train, train_csv_path)
    save_dataset(x_test, y_test, test_csv_path)

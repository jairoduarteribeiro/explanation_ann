import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import join, dirname


def load_data():
    path = join(dirname(__file__), 'heart.csv')
    df = pd.read_csv(path)
    x = df.iloc[:, :-1]
    x_columns = df.columns[:-1]
    y = df.iloc[:, -1]
    for scaler in (MinMaxScaler(), StandardScaler()):
        x = scaler.fit_transform(x)
        x = pd.DataFrame(x, columns=x_columns)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True,
                         stratify=y)
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    train_csv = pd.concat([x_train, y_train], axis=1)
    train_csv_path = join(dirname(__file__), 'train.csv')
    train_csv.to_csv(train_csv_path, index=False)
    test_csv = pd.concat([x_test, y_test], axis=1)
    test_csv_path = join(dirname(__file__), 'test.csv')
    test_csv.to_csv(test_csv_path, index=False)

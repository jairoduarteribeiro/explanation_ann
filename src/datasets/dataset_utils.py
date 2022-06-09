import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import join, dirname


def get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


def transform(x, x_columns=None):
    pipe = Pipeline(steps=[('scaler1', MinMaxScaler()), ('scaler2', StandardScaler())])
    x = pipe.fit_transform(x)
    return pd.DataFrame(x, columns=x_columns)


def split_dataset(x, y):
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True,
                         stratify=y)
    return (x_train, y_train), (x_test, y_test)


def save_dataset(x, y, csv_path):
    csv = pd.concat([x, y], axis=1)
    csv.to_csv(csv_path, index=False)


if __name__ == '__main__':
    print(get_dataset_path('heart_disease', 'heart.csv'))

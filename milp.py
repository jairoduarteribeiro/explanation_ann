import numpy as np
import pandas as pd
from os.path import join, dirname


def get_domain_and_bounds(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 2:
            domain.append('B')
        elif np.any(unique_values.astype(np.int64) !=
                    unique_values.astype(np.float64)):
            domain.append('C')
        else:
            domain.append('I')
        lower_bound = dataframe[column].min()
        upper_bound = dataframe[column].max()
        bounds.append([lower_bound, upper_bound])
    return domain, np.array(bounds)


if __name__ == '__main__':
    dataset_path = join(dirname(__file__), 'datasets', 'heart_disease',
                        'heart.csv')
    data = pd.read_csv(dataset_path)
    domain, bounds = get_domain_and_bounds(data)
    print(domain)
    print(bounds)

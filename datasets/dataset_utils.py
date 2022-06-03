from os.path import join, dirname


def get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


if __name__ == '__main__':
    print(get_dataset_path('heart_disease', 'heart.csv'))

from os.path import join, dirname


def get_model_path(*paths):
    return join(dirname(__file__), *paths)

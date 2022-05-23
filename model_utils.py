from os.path import join


def model_path(model_name):
    return join('models', f'{model_name}.h5')

import pandas as pd
from tensorflow.keras.models import load_model
from docplex.mp.model import Model
from os.path import join, dirname
from solver_utils import get_input_domain_and_bounds, get_input_variables, get_output_variables, \
    get_intermediate_variables, get_auxiliary_variables, get_decision_variables
from tjeng import build_tjeng_network
from fischetti import build_fischetti_network


def build_network(model, dataframe, method):
    layers = model.layers
    mdl = Model()
    input_domain, input_bounds = get_input_domain_and_bounds(dataframe)
    variables = {
        'input': get_input_variables(mdl, input_domain, input_bounds),
        'intermediate': [],
        'auxiliary': [],
        'decision': []
    }
    for layer_idx, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]
        if layer_idx == len(layers) - 1:
            variables['output'] = get_output_variables(mdl, number_variables)
            break
        variables['intermediate'].append(get_intermediate_variables(mdl, layer_idx, number_variables))
        if method == 'fischetti':
            variables['auxiliary'].append(get_auxiliary_variables(mdl, layer_idx, number_variables))
        variables['decision'].append(get_decision_variables(mdl, layer_idx, number_variables))
    if method == 'fischetti':
        mdl, output_bounds = build_fischetti_network(mdl, layers, variables)
    else:
        mdl, output_bounds = build_tjeng_network(mdl, layers, variables)
    return mdl, output_bounds


if __name__ == '__main__':
    train_path = join(dirname(__file__), 'datasets', 'heart_disease', 'train.csv')
    train_data = pd.read_csv(train_path)
    test_path = join(dirname(__file__), 'datasets', 'heart_disease', 'train.csv')
    test_data = pd.read_csv(test_path)
    dataframe = pd.concat([train_data, test_data], ignore_index=True)
    model_path = join(dirname(__file__), 'models', 'heart.h5')
    model = load_model(model_path)
    mdl, output_bounds = build_network(model, dataframe, 'tjeng')
    print(mdl.export_to_string())
    print(output_bounds)

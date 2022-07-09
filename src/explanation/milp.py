from docplex.mp.model import Model
from explanation.fischetti import build_fischetti_network
from explanation.solver_utils import get_auxiliary_variables, get_decision_variables, get_input_domain_and_bounds, \
    get_input_variables, get_intermediate_variables, get_output_variables


def build_network(model, dataframe):
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
        variables['auxiliary'].append(get_auxiliary_variables(mdl, layer_idx, number_variables))
        variables['decision'].append(get_decision_variables(mdl, layer_idx, number_variables))
    mdl, output_bounds = build_fischetti_network(mdl, layers, variables)
    bounds = {
        'input': input_bounds,
        'output': output_bounds
    }
    return mdl, bounds

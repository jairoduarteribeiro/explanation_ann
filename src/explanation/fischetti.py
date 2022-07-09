import numpy as np

from explanation.solver_utils import maximize, minimize


def build_fischetti_network(mdl, layers, variables):
    output_bounds = []
    number_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        x = variables['input'] if layer_idx == 0 else variables['intermediate'][layer_idx - 1]
        _A = layer.get_weights()[0].T
        _b = layer.bias.numpy()
        number_neurons = len(_A)
        if layer_idx != number_layers - 1:
            _s = variables['auxiliary'][layer_idx]
            _a = variables['decision'][layer_idx]
            _y = variables['intermediate'][layer_idx]
        else:
            _s = np.empty(number_neurons)  # prevent exception in enumerate
            _a = np.empty(number_neurons)
            _y = variables['output']
        for neuron_idx, (A, b, y, a, s) in enumerate(zip(_A, _b, _y, _a, _s)):
            result = A @ x + b
            if layer_idx != number_layers - 1:
                mdl.add_constraint(result == y - s, ctname=f'c_{layer_idx}_{neuron_idx}')
                mdl.add_indicator(a, y <= 0, 1)
                mdl.add_indicator(a, s <= 0, 0)
                upper_bound_y = maximize(mdl, y)
                upper_bound_s = maximize(mdl, s)
                y.set_ub(upper_bound_y)
                s.set_ub(upper_bound_s)
            else:
                mdl.add_constraint(result == y, ctname=f'c_{layer_idx}_{neuron_idx}')
                upper_bound = maximize(mdl, y)
                lower_bound = minimize(mdl, y)
                y.set_ub(upper_bound)
                y.set_lb(lower_bound)
                output_bounds.append([lower_bound, upper_bound])
    return mdl, output_bounds


def insert_output_constraints_fischetti(mdl, network_output, variables):
    output_variable = variables['output'][network_output]
    binary_idx = 0
    for output_idx, output in enumerate(variables['output']):
        if output_idx != network_output:
            indicator = variables['binary'][binary_idx]
            mdl.add_indicator(indicator, output_variable <= output, 1)
            binary_idx += 1

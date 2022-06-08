import numpy as np
from solver_utils import maximize, minimize


def build_tjeng_network(mdl, layers, variables):
    output_bounds = []
    number_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        x = variables['input'] if layer_idx == 0 else variables['intermediate'][layer_idx - 1]
        _A = layer.get_weights()[0].T
        _b = layer.bias.numpy()
        number_neurons = len(_A)
        if layer_idx != number_layers - 1:
            _a = variables['decision'][layer_idx]
            _y = variables['intermediate'][layer_idx]
        else:
            _a = np.empty(number_neurons)  # prevent exception in enumerate
            _y = variables['output']
        for neuron_idx, (A, b, y, a) in enumerate(zip(_A, _b, _y, _a)):
            progress = (layer_idx * number_neurons + neuron_idx) / (number_layers * number_neurons)
            print(f'Progress: {progress * 100:.2f}%')
            result = A @ x + b
            upper_bound = maximize(mdl, result)
            if upper_bound <= 0 and layer_idx != number_layers - 1:
                mdl.add_constraint(y == 0, ctname=f'c_{layer_idx}_{neuron_idx}')
                continue
            lower_bound = minimize(mdl, result)
            if lower_bound >= 0 and layer_idx != number_layers - 1:
                mdl.add_constraint(result == y, ctname=f'c_{layer_idx}_{neuron_idx}')
                continue
            if layer_idx != number_layers - 1:
                mdl.add_constraint(y <= result - lower_bound * (1 - a))
                mdl.add_constraint(y >= result)
                mdl.add_constraint(y <= upper_bound * a)
            else:
                mdl.add_constraint(result == y)
                output_bounds.append([lower_bound, upper_bound])
    print('Progress: 100.00%')
    return mdl, output_bounds


def insert_tjeng_output_constraints(mdl, output_bounds, network_output, variables):
    output_variable = variables['output'][network_output]
    diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]
    binary_idx = 0
    for output_idx, output in enumerate(variables['output']):
        if output_idx != network_output:
            diff = diffs[output_idx]
            binary_variable = variables['binary'][binary_idx]
            mdl.add_constraint(output_variable - output - diff * (1 - binary_variable) <= 0)
            binary_idx += 1

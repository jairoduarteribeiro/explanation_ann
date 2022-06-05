from solver_utils import maximize, minimize


def build_tjeng_network(mdl, layers, variables):
    output_bounds = []
    number_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        x = variables['input'] if layer_idx == 0 else variables['intermediate'][layer_idx - 1]
        _A = layer.get_weights()[0].T
        _b = layer.bias.numpy()
        if layer_idx != len(layers) - 1:
            _a = variables['decision'][layer_idx]
            _y = variables['intermediate'][layer_idx]
        else:
            _a = None
            _y = variables['output']
        number_neurons = len(_A)
        for neuron_idx, (A, b, y, a) in enumerate(zip(_A, _b, _y, _a)):
            progress = (layer_idx * number_neurons + neuron_idx) / (number_layers * number_neurons)
            print(f'Progress: {progress * 100:.2f}%')
            upper_bound = maximize(mdl, A @ x + b)
            if upper_bound <= 0 and layer_idx != len(layers) - 1:
                mdl.add_constraint(y == 0, ctname=f'c_{layer_idx}_{neuron_idx}')
                continue
            lower_bound = minimize(mdl, A @ x + b)
            if lower_bound >= 0 and layer_idx != len(layers) - 1:
                mdl.add_constraint(A @ x + b == y, ctname=f'c_{layer_idx}_{neuron_idx}')
                continue
            if layer_idx != len(layers) - 1:
                mdl.add_constraint(y <= A @ x + b - lower_bound * (1 - a))
                mdl.add_constraint(y >= A @ x + b)
                mdl.add_constraint(y <= upper_bound * a)
            else:
                mdl.add_constraint(A @ x + b == y)
                output_bounds.append([lower_bound, upper_bound])
    print('Progress: 100.00%')
    return mdl, output_bounds

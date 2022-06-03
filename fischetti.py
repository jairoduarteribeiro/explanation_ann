from solver_utils import maximize, minimize


def build_fischetti_network(mdl, layers, variables):
    output_bounds = []
    for layer_idx, layer in enumerate(layers):
        x = variables['input'] if layer_idx == 0 else variables['intermediate'][layer_idx - 1]
        _A = layer.get_weights()[0].T
        _b = layer.bias.numpy()
        if layer_idx != len(layers) - 1:
            _s = variables['auxiliary'][layer_idx]
            _a = variables['decision'][layer_idx]
            _y = variables['intermediate'][layer_idx]
        else:
            _s = None
            _a = None
            _y = variables['output']
        for idx, (A, b, y, a, s) in enumerate(zip(_A, _b, _y, _a, _s)):
            if layer_idx != len(layers) - 1:
                mdl.add_constraint(A @ x + b == y - s, ctname=f'c_{layer_idx}_{idx}')
                mdl.add_indicator(a, y <= 0, 1)
                mdl.add_indicator(a, s <= 0, 0)
                upper_bound_y = maximize(mdl, y)
                upper_bound_s = maximize(mdl, s)
                y.set_ub(upper_bound_y)
                s.set_ub(upper_bound_s)
            else:
                mdl.add_constraint(A @ x + b == y, ctname=f'c_{layer_idx}_{idx}')
                upper_bound = maximize(mdl, y)
                lower_bound = minimize(mdl, y)
                y.set_ub(upper_bound)
                y.set_lb(lower_bound)
                output_bounds.append([lower_bound, upper_bound])
    return mdl, output_bounds

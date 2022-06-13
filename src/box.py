import numpy as np


def box_sum(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def box_product(a, weight):
    if weight < 0:
        a = [a[1], a[0]]
    return [weight * a[0], weight * a[1]]


def box_relu(a):
    return [max(a[0], 0), max(a[1], 0)]


def box_forward(input_bounds, input_weights, input_biases, apply_relu=True):
    result_forward = []
    for idx, (weights, bias) in enumerate(zip(input_weights, input_biases)):
        result = [0, 0]
        bias_bounds = [bias, bias]
        for bounds, weight in zip(input_bounds, weights):
            product = box_product(bounds, weight)
            result = box_sum(result, product)
        result = box_sum(result, bias_bounds)
        if apply_relu:
            result = box_relu(result)
        result_forward.append(result)
    return np.array(result_forward)


def box_fix_input_bounds(input_bounds, network_input, input_idx):
    fixed_bounds = []
    for idx, bounds in enumerate(input_bounds):
        if idx == input_idx:
            fixed_bounds.append([bounds[0], bounds[1]])
        else:
            feature = network_input[0][idx].numpy()
            fixed_bounds.append([feature, feature])
    return np.array(fixed_bounds)


def box_check_solution(bounds, network_output):
    lower_bound = bounds[network_output][0]
    bounds = np.delete(bounds, network_output, axis=0)
    max_upper_bound = np.max(bounds, axis=0)[1]
    return lower_bound > max_upper_bound


def box_has_solution(bounds, layers, network_output):
    for layer_idx, layer in enumerate(layers):
        weights = layer.get_weights()[0].T
        biases = layer.bias.numpy()
        bounds = box_forward(bounds, weights, biases) if layer_idx != len(layers) - 1 \
            else box_forward(bounds, weights, biases, apply_relu=False)
    return box_check_solution(bounds, network_output)

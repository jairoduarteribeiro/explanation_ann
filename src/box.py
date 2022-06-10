def box_sum(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def box_product(a, weight):
    if weight < 0:
        a = [a[1], a[0]]
    return [weight * a[0], weight * a[1]]


def box_relu(a):
    return [max(a[0], 0), max(a[1], 0)]


def box_forward(input_bounds, input_weights):
    result_forward = []
    for weights in input_weights:
        result = [0, 0]
        for bounds, weight in zip(input_bounds, weights):
            product = box_product(bounds, weight)
            result = box_sum(result, product)
        result_forward.append(box_relu(result))
    return result_forward

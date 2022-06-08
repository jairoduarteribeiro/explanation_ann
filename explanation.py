from fischetti import insert_output_constraints_fischetti
from tjeng import insert_tjeng_output_constraints


def get_minimal_explanation(mdl, output_bounds, method, network_input, network_output, number_classes):
    variables = {
        'output': [mdl.get_var_by_name(f'o_{class_idx}') for class_idx in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{idx}') == feature.numpy() for idx, feature in enumerate(network_input[0])],
        names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    if method == 'tjeng':
        insert_tjeng_output_constraints(mdl, output_bounds, network_output, variables)
    else:
        insert_output_constraints_fischetti(mdl, network_output, variables)
    for constraint in input_constraints:
        mdl.remove_constraint(constraint)
        mdl.solve(log_output=True)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
    return mdl.find_matching_linear_constraints('input')

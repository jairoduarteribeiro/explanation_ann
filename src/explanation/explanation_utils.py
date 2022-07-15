import tensorflow as tf

from explanation.box import box_has_solution, box_relax_input_bounds
from explanation.fischetti import insert_fischetti_output_constraints
from explanation.tjeng import insert_tjeng_output_constraints


def log_explanation(logger, features, explanation):
    def constraint_to_idx(constraint):
        return int(constraint.lhs.name[2:])
    columns = [constraint_to_idx(constraint) for constraint in explanation]
    result = list(features[columns])
    logger.info(f'Explanation: {result}')


def get_minimal_explanation(mdl, bounds, method, network_input, network_output, layers, using_box=False):
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{class_idx}') for class_idx in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{idx}') == feature.numpy() for idx, feature in enumerate(network_input[0])],
        names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    if method == 'fischetti':
        insert_fischetti_output_constraints(mdl, network_output, variables)
    else:
        insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    relax_idx = []
    for input_idx, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        if using_box:
            relax_idx.append(input_idx)
            input_bounds = box_relax_input_bounds(bounds['input'], network_input, relax_idx)
            if box_has_solution(input_bounds, layers, network_output):
                print(f'Feature {input_idx} is not relevant (using box)')
                continue
            relax_idx.pop()
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            print(f'Feature {input_idx} is relevant (using solver)')
            continue
        print(f'Feature {input_idx} is not relevant (using solver)')
    return mdl.find_matching_linear_constraints('input')

from fischetti import insert_output_constraints_fischetti
from tjeng import insert_tjeng_output_constraints
from box import box_fix_input_bounds, box_has_solution


def print_explanation(x_idx, features, explanation):
    def constraint_to_idx(constraint):
        return int(constraint.lhs.name[2:])
    columns = [constraint_to_idx(constraint) for constraint in explanation]
    result = list(features[columns])
    print(f'Explanation for data {x_idx}: {result}')


def get_minimal_explanation(mdl, bounds, method, network_input, network_output, layers):
    number_classes = len(bounds['output'])
    variables = {
        'output': [mdl.get_var_by_name(f'o_{class_idx}') for class_idx in range(number_classes)],
        'binary': mdl.binary_var_list(number_classes - 1, name='b')
    }
    input_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{idx}') == feature.numpy() for idx, feature in enumerate(network_input[0])],
        names='input')
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    if method == 'tjeng':
        insert_tjeng_output_constraints(mdl, bounds['output'], network_output, variables)
    else:
        insert_output_constraints_fischetti(mdl, network_output, variables)
    for input_idx, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        input_bounds = box_fix_input_bounds(bounds['input'], network_input, input_idx)
        if box_has_solution(input_bounds, layers, network_output):
            print(f'Feature {input_idx} is not relevant (using box)')
            continue
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            print(f'Feature {input_idx} is relevant (using solver)')
            continue
        print(f'Feature {input_idx} is not relevant (using solver)')
    return mdl.find_matching_linear_constraints('input')

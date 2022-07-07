import numpy as np


def get_input_domain_and_bounds(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 2:
            domain.append('B')
        elif np.any(unique_values.astype(np.int64) != unique_values.astype(np.float64)):
            domain.append('C')
        else:
            domain.append('I')
        lower_bound = dataframe[column].min()
        upper_bound = dataframe[column].max()
        bounds.append([lower_bound, upper_bound])
    return domain, bounds


def get_input_variables(mdl, input_domain, input_bounds):
    input_variables = []
    for idx, (domain, bounds) in enumerate(zip(input_domain, input_bounds)):
        lower_bound, upper_bound = bounds
        name = f'x_{idx}'
        if domain == 'C':
            input_variables.append(mdl.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
        elif domain == 'I':
            input_variables.append(mdl.integer_var(lb=lower_bound, ub=upper_bound, name=name))
        else:
            input_variables.append(mdl.binary_var(name=name))
    return input_variables

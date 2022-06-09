import unittest
import pandas as pd
import numpy as np
from docplex.mp.model import Model
from solver_utils import get_input_domain_and_bounds, get_input_variables


class TestSolverUtils(unittest.TestCase):
    def test_get_input_domain_and_bounds(self):
        dataframe = pd.DataFrame({
            'age': [18, 19, 20],
            'weight': [65.9, 58.0, 89.5],
            'gender': [0, 1, 0],
            'target': [0, 1, 2]
        })
        domain, bounds = get_input_domain_and_bounds(dataframe)
        self.assertEqual(domain, ['I', 'C', 'B'])
        self.assertTrue(np.array_equal(bounds, np.array([[18, 20], [58.0, 89.5], [0, 1]])))

    def test_get_input_variables(self):
        mdl = Model()
        domain = ['I', 'C', 'B']
        bounds = np.array([[18, 20], [58.0, 89.5], [0, 1]])
        input_variables = get_input_variables(mdl, domain, bounds)
        self.assertTrue(input_variables[0].is_integer())
        self.assertEqual(input_variables[0].name, 'x_0')
        self.assertEqual(input_variables[0].lb, bounds[0][0])
        self.assertEqual(input_variables[0].ub, bounds[0][1])
        self.assertTrue(input_variables[1].is_continuous())
        self.assertEqual(input_variables[1].name, 'x_1')
        self.assertEqual(input_variables[1].lb, bounds[1][0])
        self.assertEqual(input_variables[1].ub, bounds[1][1])
        self.assertTrue(input_variables[2].is_binary())
        self.assertEqual(input_variables[2].name, 'x_2')
        self.assertEqual(input_variables[2].lb, bounds[2][0])
        self.assertEqual(input_variables[2].ub, bounds[2][1])


if __name__ == '__main__':
    unittest.main()

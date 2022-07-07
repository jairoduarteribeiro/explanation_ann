import unittest
import pandas as pd

from docplex.mp.model import Model
from explanation.solver_utils import get_auxiliary_variables, get_input_domain_and_bounds, get_input_variables, \
    get_intermediate_variables


class TestSolverUtils(unittest.TestCase):
    def test_get_input_domain_and_bounds(self):
        dataframe = pd.DataFrame({
            'age': [18, 19, 20],
            'weight': [65.9, 58.0, 89.5],
            'gender': [0, 1, 0],
            'target': [0, 1, 2]
        })
        domain, bounds = get_input_domain_and_bounds(dataframe)
        self.assertEquals(domain, ['I', 'C', 'B'])
        self.assertEquals(bounds, [[18, 20], [58.0, 89.5], [0, 1]])

    def test_get_input_variables(self):
        mdl = Model()
        domain = ['I', 'C', 'B']
        bounds = [[18, 20], [58.0, 89.5], [0, 1]]
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

    def test_get_intermediate_variables(self):
        mdl = Model()
        number_variables = 3
        intermediate_variables = get_intermediate_variables(mdl, 1, number_variables)
        self.assertEqual(len(intermediate_variables), number_variables)
        self.assertTrue(intermediate_variables[0].is_continuous())
        self.assertEqual(intermediate_variables[0].name, 'y_1_0')
        self.assertEqual(intermediate_variables[0].lb, 0)
        self.assertEqual(intermediate_variables[0].ub, mdl.infinity)
        self.assertTrue(intermediate_variables[1].is_continuous())
        self.assertEqual(intermediate_variables[1].name, 'y_1_1')
        self.assertEqual(intermediate_variables[1].lb, 0)
        self.assertEqual(intermediate_variables[1].ub, mdl.infinity)
        self.assertTrue(intermediate_variables[2].is_continuous())
        self.assertEqual(intermediate_variables[2].name, 'y_1_2')
        self.assertEqual(intermediate_variables[2].lb, 0)
        self.assertEqual(intermediate_variables[2].ub, mdl.infinity)

    def test_get_auxiliary_variables(self):
        mdl = Model()
        number_variables = 3
        auxiliary_variables = get_auxiliary_variables(mdl, 1, number_variables)
        self.assertEqual(len(auxiliary_variables), number_variables)
        self.assertTrue(auxiliary_variables[0].is_continuous())
        self.assertEqual(auxiliary_variables[0].name, 's_1_0')
        self.assertEqual(auxiliary_variables[0].lb, 0)
        self.assertEqual(auxiliary_variables[0].ub, mdl.infinity)
        self.assertTrue(auxiliary_variables[1].is_continuous())
        self.assertEqual(auxiliary_variables[1].name, 's_1_1')
        self.assertEqual(auxiliary_variables[1].lb, 0)
        self.assertEqual(auxiliary_variables[1].ub, mdl.infinity)
        self.assertTrue(auxiliary_variables[2].is_continuous())
        self.assertEqual(auxiliary_variables[2].name, 's_1_2')
        self.assertEqual(auxiliary_variables[2].lb, 0)
        self.assertEqual(auxiliary_variables[2].ub, mdl.infinity)


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd

from explanation.solver_utils import get_input_domain_and_bounds


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


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from box import box_sum, box_product, box_relu, box_forward


class TestBox(unittest.TestCase):
    def test_box_sum(self):
        a = [1, 2]
        b = [3, 4]
        result = box_sum(a, b)
        self.assertEqual(result, [4, 6])

    def test_box_product(self):
        a = [1, 2]
        weight1 = 3
        weight2 = -4
        result1 = box_product(a, weight1)
        result2 = box_product(a, weight2)
        self.assertEqual(result1, [3, 6])
        self.assertEqual(result2, [-8, -4])

    def test_box_relu(self):
        a = [1, -2]
        b = [-3, 4]
        relu_a = box_relu(a)
        relu_b = box_relu(b)
        self.assertEqual(relu_a, [1, 0])
        self.assertEqual(relu_b, [0, 4])

    def test_box_forward(self):
        input_bounds = [[0, 0.3], [0.1, 0.4]]
        input_weights = np.array([[1, 1], [1, -1]]).T
        result = box_forward(input_bounds, input_weights)
        assert_array_almost_equal(result, [[0.1, 0.7], [0, 0.2]])


if __name__ == '__main__':
    unittest.main()

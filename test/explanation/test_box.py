import unittest
import tensorflow as tf

from numpy.testing import assert_array_almost_equal
from explanation.box import box_check_solution, box_fix_input_bounds, box_forward, box_product, box_relu, box_sum


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
        result1 = box_product(weight1, a)
        result2 = box_product(weight2, a)
        self.assertEqual(result1, [3, 6])
        self.assertEqual(result2, [-8, -4])

    def test_box_relu(self):
        a = [1, -2]
        b = [-3, 4]
        c = [1, 2]
        d = [-3, -4]
        relu_a = box_relu(a)
        relu_b = box_relu(b)
        relu_c = box_relu(c)
        relu_d = box_relu(d)
        self.assertEqual(relu_a, [1, 0])
        self.assertEqual(relu_b, [0, 4])
        self.assertEqual(relu_c, [1, 2])
        self.assertEqual(relu_d, [0, 0])

    def test_box_forward(self):
        input_bounds = [[0, 0.3], [0.1, 0.4]]
        input_weights = [[1, 1], [1, -1]]
        input_biases = [0, 0]
        output_bounds = box_forward(input_bounds, input_weights, input_biases)
        assert_array_almost_equal(output_bounds, [[0.1, 0.7], [0, 0.2]])
        input_bounds = [[0.1, 0.7], [0, 0.2]]
        input_weights = [[1, 1], [1, -1]]
        input_biases = [0.5, -0.5]
        output_bounds = box_forward(input_bounds, input_weights, input_biases, apply_relu=False)
        assert_array_almost_equal(output_bounds, [[0.6, 1.4], [-0.6, 0.2]])

    def test_box_fix_input_bounds(self):
        input_bounds = [[1, 2], [3, 4], [5, 6]]
        network_input = [[1.5, 3.5, 5.5]]
        input_idx = 1
        fixed_bounds = box_fix_input_bounds(input_bounds, network_input, input_idx)
        assert_array_almost_equal(fixed_bounds, [[1.5, 1.5], [3, 4], [5.5, 5.5]])

    def test_box_check_solution(self):
        bounds = [[0.6, 1.4], [-0.6, 0.2]]
        network_output = 0
        self.assertTrue(box_check_solution(bounds, network_output))
        bounds = [[0.4, 1.4], [-0.6, 0.4]]
        network_output = 0
        self.assertFalse(box_check_solution(bounds, network_output))


if __name__ == '__main__':
    unittest.main()

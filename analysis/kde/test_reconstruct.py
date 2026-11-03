import unittest
import numpy as np
from reconstruct import gaussian_density


class GaussianDensityTest(unittest.TestCase):
    def test_integrates_to_one_at_smallest_bandwidth(self):
        axis = np.arange(-300, 301, 25, dtype=float)
        values = gaussian_density(np.array([[7.3, -11.2]]), axis, axis, 35.8)
        self.assertAlmostEqual(float(values.sum() * 25**2), 1, places=5)

    def test_matches_direct_gaussian_formula(self):
        xs = np.arange(-100, 101, 25, dtype=float)
        ys = np.arange(-75, 76, 25, dtype=float)
        points = np.array([[7.3, -11.2], [30.1, 14.9]])
        actual = gaussian_density(points, xs, ys, 35.8)
        xx, yy = np.meshgrid(xs, ys)
        expected = sum(np.exp(-((xx-x)**2+(yy-y)**2)/(2*35.8**2))
                       for x, y in points) / (len(points)*2*np.pi*35.8**2)
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_duplicate_samples_do_not_double_probability_density(self):
        axis = np.arange(-100, 101, 25, dtype=float)
        single = gaussian_density(np.array([[0., 0.]]), axis, axis, 100.)
        double = gaussian_density(np.array([[0., 0.], [0., 0.]]), axis, axis, 100.)
        np.testing.assert_allclose(single, double)


if __name__ == '__main__':
    unittest.main()

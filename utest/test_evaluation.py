import unittest

from common.evaluation import ClassificationEvaluator
import numpy as np


class ClassificationEvaluatorTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.__estimated_labels = np.array([['A', 'A', 'B', 'A', 'B' ]]).T
        self.__gt_labels = np.array([['A', 'C', 'B', 'A', 'B' ]]).T

    def test_error_rate(self):
        
        class_eval = ClassificationEvaluator(self.__estimated_labels, self.__gt_labels)
        result = class_eval.error_rate()
        err_rate_ref = 20.0
        n_wrong_ref = 1
        n_samples_ref = self.__gt_labels.shape[0]
        result_ref = (err_rate_ref, n_wrong_ref, n_samples_ref)
        self.assertEqual(result, result_ref)
        self.assertEqual(result[0], 100 * float(result[1]) / result[2])
        
        mask_arr = np.array([[False, True, False, True, False]]).T
        result = class_eval.error_rate(mask=mask_arr)
        err_rate_ref = 50.0
        n_wrong_ref = 1
        n_samples_ref = 2
        result_ref = (err_rate_ref, n_wrong_ref, n_samples_ref)
        self.assertEqual(result, result_ref)

    def test_category_error_rates(self):
        class_eval = ClassificationEvaluator(self.__estimated_labels, self.__gt_labels)
        result = class_eval.category_error_rates()
        result_ref = [('A', 0.0, 0, 2), ('B', 0.0, 0, 2), ('C', 100.0, 1, 1)]
        self.assertEqual(result, result_ref)


if __name__ == "__main__":
    unittest.main()

import unittest
from skltemplate import IfnClassifier
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
import warnings


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class IfnTestCases(unittest.TestCase):

    def testaaa(self):
        x =3


    @ignore_warnings
    def testNetworkBuilt(self):

        x = [[0, 0, 2, 1], [0, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 2], [2, 1, 2, 1], [0, 2, 1, 0],
             [2, 0, 2, 2], [0, 1, 0, 1], [2, 1, 1, 0]]
        y = [0, 2, 1, 0, 1, 0, 0, 0, 2, 1]

        clf = IfnClassifier()
        clf.fit(x, y)

        self.assertIsNotNone(clf.network.root_node)

    @ignore_warnings
    def testClassLayer(self):
        iris = load_iris()
        digits = load_digits()
        breast_cancer = load_breast_cancer()
        wine = load_wine()

        clf = IfnClassifier()

        # test number of classes in iris
        # clf.fit(iris.data, iris.target)
        # self.assertEqual(3, len(clf.network.target_layer))

        # test number of classes in digits
        # clf.fit(digits.data, digits.target)
        # self.assertEqual(13, len(clf.network.target_layer))

        # test number of classes in breast_cancer
        clf.fit(breast_cancer.data, breast_cancer.target)
        self.assertEqual(2, len(clf.network.target_layer))

        # test number of classes in wine
        # clf.fit(wine.data, wine.target)
        # self.assertEqual(13, len(clf.network.target_layer))


if __name__ == '__main__':
    unittest.main()

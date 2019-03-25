import pytest
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from skltemplate import IfnClassifier

@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_estimator(data):
    clf = IfnClassifier()

    clf.fit(*data)
    assert hasattr(clf, 'is_fitted_') # hasattr means- as attribute.

    X = data[0]
    y_pred = clf.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_class_layer(data):
    clf = IfnClassifier()

    # test number of classes in iris
    clf.fit(*data)
    assert 3 == len(clf.network.target_layer)

    # test number of classes in digits
    # clf.fit(digits.data, digits.target)
    # self.assertEqual(13, len(clf.network.target_layer))

    # test number of classes in breast_cancer
    # clf.fit(breast_cancer.data, breast_cancer.target)
    # self.assertEqual(2, len(clf.network.target_layer))

    # test number of classes in wine
    # clf.fit(wine.data, wine.target)


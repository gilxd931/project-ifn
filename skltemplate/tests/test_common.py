import pytest

from sklearn.utils.estimator_checks import check_estimator

from skltemplate import _ifnClassifier
from skltemplate import TemplateClassifier
from skltemplate import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [_ifnClassifier, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

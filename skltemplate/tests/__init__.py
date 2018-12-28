
from skltemplate import IfnClassifier
from sklearn.datasets import load_iris

iris = load_iris()

clf = IfnClassifier(iris.data, iris.target)


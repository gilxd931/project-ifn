
from skltemplate import IfnClassifier
from sklearn import tree

# clf = IfnClassifier()

# X = [[0, 0], [1, 1]]
# Y = [0, 1]
clf = IfnClassifier()
# clf = clf.fit(X, Y)


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# iris = load_iris()
#
# print(iris)
x = [[0,0,0,1], [0,1,0,1], [1,1,0,0], [1,0,0,1], [0,0,0,0], [0,1,0,1]]
y = [0,1,1,0,1,0]
clf.fit(x, y)




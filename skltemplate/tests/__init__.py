
from skltemplate import IfnClassifier
from sklearn.datasets import load_iris
import pandas as pd

dt= pd.read_csv("test_dataset_1.csv", encoding='latin1')

print (dt)
iris = load_iris()

x = [[1,1,0], [0,1,1], [1,0,1], [0,1,0] ]
y= [1, 0 , 0, 1]
clf = IfnClassifier()
clf.fit(x, y)
x=3

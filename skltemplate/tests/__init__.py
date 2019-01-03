
from skltemplate import IfnClassifier
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.metrics import mutual_info_score


# clf = IfnClassifier()


x = [[0,0,0,1], [0,1,0,1], [1,1,0,0], [1,0,0,1], [0,0,0,0], [0,1,0,1]]
y = [0,1,1,0,1,0]

clf = IfnClassifier()

clf.fit(x, y)



# x= [[ 0,0,0,1], []]
#
# # x = [[1,1,0], [0,1,1], [1,       [0,1,0] ]
# # y= [1, 0 , 0, 1]          
# # x=3

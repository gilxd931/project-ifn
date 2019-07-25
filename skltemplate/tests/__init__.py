from skltemplate import IfnClassifier
from skltemplate import _csvConveter

clf = IfnClassifier(0.999)
x = []


# CREDIT

data = _csvConveter.CsvConverter.convert("glass.csv")

clf.fit(data[0], data[1], data[2])
clf.add_training_set_error_rate(data[0], data[1])

clf.network.create_network_structure_file()

# -------------- predict will return the classes and write it to file --------------

# print(clf.predict([[0, 2, 5, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#                          [1, 41, 5, 3, 0, 1, 2, 0, 0, 14, 1, 0, 500, 159]]))

# -------------- predict_proba will return the probability for every class and write it to file --------------
print(clf.predict_proba([[0, 2, 5, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                         [1, 41, 5, 3, 0, 1, 2, 0, 0, 14, 1, 0, 500, 159]]))


from skltemplate import IfnClassifier
from skltemplate import _csvConveter

clf = IfnClassifier(0.999)
x = []


# CREDIT

data = _csvConveter.CsvConverter.convert("glass.csv")
# df = pd.read_csv("credit_full.csv")
#
# cols = list(df.columns.values)
# y = df['Class'].values
# df = df.drop("Class", axis=1)
# for index, row in df.iterrows():
#     record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]]
#     x.append(record)


# GLASS

# dfg = pd.read_csv("glass.csv")
# y = []
# for index, row in dfg.iterrows():
#     record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
#     x.append(record)
#     y.append(row[9])



# CHESS
# y= []
# df = pd.read_csv("chess.csv")
# for index, row in df.iterrows():
    # record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[21], row[23], row[24], row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35]]
    # x.append(record)
    # y.append(row[36])

clf.fit(data[0], data[1], data[2])
clf.add_training_set_error_rate(data[0], data[1])

clf.network.create_network_structure_file()

# -------------- predict will return the classes and write it to file --------------

# print(clf.predict([[0, 2, 5, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#                          [1, 41, 5, 3, 0, 1, 2, 0, 0, 14, 1, 0, 500, 159]]))

# -------------- predict_proba will return the probability for every class and write it to file --------------
print(clf.predict_proba([[0, 2, 5, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                         [1, 41, 5, 3, 0, 1, 2, 0, 0, 14, 1, 0, 500, 159]]))


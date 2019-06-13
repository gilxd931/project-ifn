
from skltemplate import IfnClassifier

clf = IfnClassifier(0.999)
x = []

import pandas as pd

df = pd.read_csv("credit_full.csv")


# y = df['Class'].values
#
# df = df.drop("Class", axis=1)
#
# # #
# for index, row in df.iterrows():
#     record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]]
#     x.append(record)


# dfg = pd.read_csv("glass.csv")
# y = []
# for index, row in dfg.iterrows():
#     record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
#     x.append(record)
#     y.append(row[9])

y= []
df = pd.read_csv("chess.csv")
for index, row in df.iterrows():
    record = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[21], row[23], row[24], row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35]]
    x.append(record)
    y.append(row[36])

# import time
# start = time. time()
clf.fit(x, y)
# end = time. time()
# print(end - start)
clf.network.create_network_structure_file()
clf.add_training_set_error_rate(x, y)
#
# print(clf.predict_proba([[0, 2, 5, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#                          [1, 41, 5, 3, 0, 1, 2, 0, 0, 14, 1, 0, 500, 159]]))
#
#
# import matplotlib.pyplot as plt
# import networkx as nx
# import random
#
#
# def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
#
#     '''
#     From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
#     Licensed under Creative Commons Attribution-Share Alike
#
#     If the graph is a tree this will return the positions to plot this in a
#     hierarchical layout.
#
#     G: the graph (must be a tree)
#
#     root: the root node of current branch
#     - if the tree is directed and this is not given,
#       the root will be found and used
#     - if the tree is directed and this is given, then
#       the positions will be just for the descendants of this node.
#     - if the tree is undirected and not given,
#       then a random choice will be used.
#
#     width: horizontal space allocated for this branch - avoids overlap with other branches
#
#     vert_gap: gap between levels of hierarchy
#
#     vert_loc: vertical location of root
#
#     xcenter: horizontal location of root
#     '''
#     if not nx.is_tree(G):
#         raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
#
#     if root is None:
#         if isinstance(G, nx.DiGraph):
#             root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
#         else:
#             root = random.choice(list(G.nodes))
#
#     def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
#         '''
#         see hierarchy_pos docstring for most arguments
#
#         pos: a dict saying where all nodes go if they have been assigned
#         parent: parent of this branch. - only affects it if non-directed
#
#         '''
#
#         if pos is None:
#             pos = {root:(xcenter,vert_loc)}
#         else:
#             pos[root] = (xcenter, vert_loc)
#         children = list(G.neighbors(root))
#         if not isinstance(G, nx.DiGraph) and parent is not None:
#             children.remove(parent)
#         if len(children)!=0:
#             dx = width/len(children)
#             nextx = xcenter - width/2 - dx/2
#             for child in children:
#                 nextx += dx
#                 pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
#                                     vert_loc = vert_loc-vert_gap, xcenter=nextx,
#                                     pos=pos, parent = root)
#         return pos
#
#
#     return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
#
# G=nx.DiGraph()
#
# G.add_node(0)
#
# add_edges_from_list = []
#
# curr_layer = clf.network.root_node.first_layer
#
#
#
# for node in clf.network.root_node.first_layer.nodes:
#     G.add_node(node.index)
#     tuple = (0, node.index)
#     add_edges_from_list.append(tuple)
#
# curr_layer = clf.network.root_node.first_layer.next_layer
# prev_layer = clf.network.root_node.first_layer
#
# while curr_layer != None:
#     for node in prev_layer.nodes:
#         if not node.is_terminal:
#             for curr_node in curr_layer.nodes:
#                 G.add_node(curr_node.index)
#                 tuple = (node.index, curr_node.index)
#                 add_edges_from_list.append(tuple)
#     prev_layer = curr_layer
#     curr_layer = curr_layer.next_layer
#
# G.add_edges_from(add_edges_from_list)
# pos = hierarchy_pos(G, 0)
# nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
# nx.draw(G, pos=pos, with_labels=True)
# plt.savefig('hierarchy.png')
# plt.show()
#


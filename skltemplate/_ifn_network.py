import math

class Node:
    def __init__(self, index):
        self.index = index

    def print_index(self):
        print(self.index)


class RootNode(Node):
    def __init__(self):
        super().__init__(-1)
        self.first_layer = None

    def set_layer(self, layer):
        self.first_layer = layer


class ClassNode(Node):
    def __init__(self, index):
        super().__init__(index)


class AttributeNode(Node):
    def __init__(self, index, layer, is_terminal=False):
        super().__init__(index)
        self.layer = layer
        self.is_terminal = is_terminal
        self.weight = math.inf

    def set_terminal(self):
        self.is_terminal = True

    def set_weight(self, weight):
        if self.is_terminal:
            self.weight = weight

    def print_info(self):
        print('')
        print('terminal: ' + str(self.is_terminal))
        print('weight: ' + str(self.weight))
        print('index: ' + str(self.index))
        print('------------------------')


class HiddenLayer:
    def __init__(self, index):
        self.index = index
        self.next_layer = None
        self.nodes = None

    def set_nodes(self, nodes):
        self.nodes = nodes

    def print(self):
        for node in self.nodes:
            node.print_index()

    def get_node(self, index):
        for node in self.nodes:
            if node.index == index:
                return node
        return None

class IfnNetwork:
    def __init__(self):
        self.target_layer = []
        self.root_node = RootNode

    def build_target_layer(self, num_of_classes):
        if len(num_of_classes) != 0:
            for i in num_of_classes:
                self.target_layer.append(ClassNode(i))

    def print_classes(self):
        for node in self.target_layer:
            node.print_index()






class Node:
    def __init__(self, index):
        self.index = index


class RootNode(Node):
    def __init__(self):
        super().__init__(-1)
        self.first_layer = Attribute_layer(None)

    def set_layer(self, layer):
        self.first_layer = layer


class ClassNode(Node):
    def __init__(self, index):
        super().__init__(index)


class AttributeNode(Node):
    def __init__(self, index, is_terminal=False):
        super().__init__(index)
        self.is_terminal = is_terminal
        self.next = []

    def add_next(self, next_index):
        exist = False
        for node in self.next:
            if next_index == node.index:
                exist = True

        if not exist:
            self.next.append(AttributeNode(next_index))

    def print_index(self):
        print(self.index)

    def print_next(self):
        for node in self.next:
            print(node.index)


class Attribute_layer:
    def __init__(self, index):
        self.index = index
        self.next_layer = None
        self.nodes = None

    def set_nodes(self, nodes):
        self.nodes = nodes


class IfnNetwork:
    def __init__(self):
        self.classes = None
        self.root_node = RootNode()

    def print_classes(self):
        print(self.root_node.first_layer.nodes)

    def build_classes(self):
        if self.classes is None:
            self.classes = [ClassNode(0), ClassNode(1)]



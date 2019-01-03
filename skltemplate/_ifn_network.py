
class Node:
    def __init__(self, index):
        self.index = index


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


    def print_next(self):
        for node in self.next:
            print(node.index)


class IfnNetwork:
    def __init__(self):
        self.classes = None
        self.root_node = AttributeNode(-1)


    def print_classes(self):
        print(self.classes)

    def print_root(self):
        print(self.root_node.index)

    def build_classes(self):
        if self.classes is None:
            self.classes = [ClassNode(0), ClassNode(1)]


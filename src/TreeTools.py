# (value, subtrees)
class TreeTools:
    def __init__(self):
        # memoization for _count_nodes functions
        self._count_nodes_dict = {}

    # Return tree is leave or not
    @staticmethod
    def _is_not_leave(tree):
        return type(tree[1]) == list

    def get_subtrees(self, tree):
        yield tree
        if self._is_not_leave(tree):
            for subtree in tree[1]:
                if self._is_not_leave(subtree):
                    for x in self.get_subtrees(subtree):
                        yield x

    # Returns pairs of paths and values of a tree
    def get_paths(self, tree):
        for i, subtree in enumerate(tree[1]):
            yield [i], subtree[0]
            if self._is_not_leave(subtree):
                for path, value in self.get_paths(subtree):
                    yield [i] + path, value

    # Returns the number of nodes in a tree (not including root)
    def count_nodes(self, tree):
        return self._count_nodes(tree[1])

    def _count_nodes(self, branches):
        if id(branches) in self._count_nodes_dict:
            return self._count_nodes_dict[id(branches)]
        size = 0
        for node in branches:
            if self._is_not_leave(node):
                size += 1 + self._count_nodes(node[1])
        self._count_nodes_dict[id(branches)] = size
        return size

    # Returns all the nodes in a path
    def get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        for decision in path:
            nodes.append(next_node)
            if not self._is_not_leave(tree):
                break
            next_node += 1 + self._count_nodes(tree[1][:decision])
            tree = tree[1][decision]
        return nodes

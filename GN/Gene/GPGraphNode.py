'''

@author: ben
'''

from __future__ import division
from GPNode import GPNode


class GPGraphNode(GPNode):

    def __init__(self, operation, children):
        self._funct = operation.get_func()
        self._name = operation.get_name()
        self._children = children
        self._arity = operation.get_arity()
        self._op = operation

    def evaluate(self, _input):
        pre_result = [child.evaluate(_input) for child in self._children]
        y_hat = self._funct(pre_result)
        return y_hat

    def __repr__(self):
        res = ''
        for child in self._children:
            res += child.__repr__()  # str(child)

        res += ' ' + self._name + ' '
        return res

    def display(self, indent=0):
        print (' ' * indent) + self._name
        for child in self._children:
            child.display(indent + 2)

    def nodes(self):
        res = []
        for child in self._children:
            res += child.nodes()
        res += [self._op]

        return res

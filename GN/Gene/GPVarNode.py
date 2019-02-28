'''

@author: ben
'''

from __future__ import division
from GPNode import GPNode


class GPVarNode(GPNode):

    def __init__(self, index):
        self._paramIndex = index

    def evaluate(self, _input):
        return _input[self._paramIndex]

    def display(self, indent=0):
        print '%sX%d' % (' ' * indent, self._paramIndex)

    def __repr__(self):
        return ' X' + str(self._paramIndex) + ' '

    def nodes(self):
        return ['X' + str(self._paramIndex)]

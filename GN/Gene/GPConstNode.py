'''

@author: ben
'''


from __future__ import division
from GPNode import GPNode


class GPConstNode(GPNode):

    def __init__(self, value):
        self._constValue = value

    def evaluate(self, _input):
        return self._constValue

    def display(self, indent=0):
        print '%s%d' % (' ' * indent, self._constValue)

    def __repr__(self):
        return ' ' + str(self._constValue) + ' '

    def nodes(self):
        return [str(self._constValue)]

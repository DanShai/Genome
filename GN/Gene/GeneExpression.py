'''

@author: dan
'''

from GOperation import GOperation
from Stack import Stack


class GeneExpression:

    def convertToExpression(self, agraph):
        nodes = agraph.nodes()
        mstack = Stack()
        for nd in nodes:
            if isinstance(nd, GOperation):
                self.format_expression(nd, mstack)
            else:
                mstack.push(str(nd))

        if mstack.isEmpty():
            _expression = 'Empty'
        else:
            res = mstack.pop()
            _expression = str(res)

        return _expression

    def format_expression(self, nd, mstack):

        arity = nd._arity
        name = nd._name
        if arity == 1:
            elem = mstack.pop()
            if name == "!":
                nexpr = '(' + str(elem) + name + ') '
            else:
                nexpr = name + '(' + str(elem) + ') '

        elif arity == 2:
            ri = mstack.pop()
            le = mstack.pop()

            if name in ['/', '*', '+', '-', '^', '&', '|', '>', '<', '%']:
                nexpr = ' (' + str(le) + name + str(ri) + ') '
            else:
                nexpr = name + '(' + str(le) + ',' + str(ri) + ') '

        elif arity == 3:
            ri = mstack.pop()
            md = mstack.pop()
            le = mstack.pop()
            nexpr = name + '(' + str(le) + ') then (' + \
                str(md) + ') else (' + str(ri) + ') '

        mstack.push(nexpr)

'''

@author: ben
'''

from __future__ import division
from GOperation import GOperation
import GFunctions as gf


class GOperationsDef:
    def __init__(self):
        self.add_o = GOperation(gf._add, 2, '+')
        self.sub_o = GOperation(gf._sub, 2, '-')
        self.mul_o = GOperation(gf._mul, 2, '*')
        self.mod_o = GOperation(gf._mod, 2, '%')
        self.div_o = GOperation(gf._div, 2, '/')
        self.sqrt_o = GOperation(gf._sqrt, 1, 'sqrt')
        self.abs_o = GOperation(gf._abs, 1, 'abs')

        self.or_o = GOperation(gf._or, 2, '|')
        self.and_o = GOperation(gf._and, 2, '&')
        self.xor_o = GOperation(gf._xor, 2, '^')
        self.gt_o = GOperation(gf._gt, 2, '>')
        self.lt_o = GOperation(gf._lt, 2, '<')
        self.if_o = GOperation(gf._if, 3, 'if')
        self.max_o = GOperation(gf._max, 2, 'max')
        self.min_o = GOperation(gf._min, 2, 'min')
        self.not_o = GOperation(gf._not, 1, '~')

        self.sin_o = GOperation(gf._sin, 1, 'sin')
        self.cos_o = GOperation(gf._cos, 1, 'cos')
        self.exp_o = GOperation(gf._exp, 1, 'exp')
        self.pow_o = GOperation(gf._pow, 2, 'pow')
        self.log_o = GOperation(gf._log, 1, 'log')
        self.inv_o = GOperation(gf._inv, 1, 'inv')

        self._FunctSlots = {
            1: [self.mul_o, self.add_o],
            2: [self.mul_o, self.sub_o, self.add_o, self.log_o, self.sqrt_o, self.exp_o, self.cos_o, self.sin_o],
            3: [self.inv_o, self.abs_o, self.div_o, self.mul_o, self.sub_o, self.add_o, self.log_o, self.sqrt_o, self.exp_o, self.cos_o, self.sin_o],
            4: [self.max_o, self.min_o, self.if_o, self.lt_o, self.gt_o, self.not_o, self.xor_o, self.and_o, self.or_o],
            5: [self.inv_o, self.div_o, self.mod_o, self.xor_o, self.and_o, self.or_o, self.mul_o, self.sub_o, self.add_o,
                self.log_o, self.sqrt_o, self.exp_o, self.cos_o, self.sin_o, self.max_o, self.min_o, self.abs_o, self.if_o, self.gt_o, self.lt_o, self.not_o]
        }

    def get_Functions(self, index):
        return self._FunctSlots[index]

    def clear_FunctSlots(self):
        self._FunctSlots[:] = []

    def getFuncSlots(self):
        return self._FunctSlots

    def add_Operation_To_Slot(self, index, f):
        self._FunctSlots[index].append(f)

    def get_cops_dict(self):
        flist = self.get_Functions(4)
        ops = [(str(x.get_name()), x) for x in flist]
        dops = dict(ops)
        return dops

    def get_ops_dict(self):

        flist = self.get_Functions(4)
        ops = [(str(x.get_name()), x.get_arity()) for x in flist]
        dops = dict(ops)
        #print "dicops : "  , dops
        return dops

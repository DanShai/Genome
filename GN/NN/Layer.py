'''

@author: dan
'''

from __future__ import division

from copy import deepcopy

import numpy as np

from GN.Gene.Genome import Genome


class Layer:
    def __init__(self, num_nodes, n_opts):
        self._num_nodes = num_nodes
        self._nOpts = n_opts
        self._nodes = np.empty(shape=(self._num_nodes,), dtype=np.object)

    def create(self):
        for i in xrange(self._num_nodes):
            n_ero = Genome(ops=self._nOpts)
            n_ero.create()
            self._nodes[i] = n_ero

    def evaluate(self, inp):
        EV = np.array([nd.evaluate(inp) for nd in self._nodes])
        return EV

    def __repr__(self):
        nodes = " "
        for nd in self._nodes:
            nodes += "[ " + nd.__repr__() + " ] \n"

        return nodes

    def get_indexes(self):
        indexes = []
        for nd in self._nodes:
            indexes += nd.get_index()

        #indexes = list(set(indexes))
        return indexes

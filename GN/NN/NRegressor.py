'''

@author: dan
'''

from __future__ import division

from copy import deepcopy

import numpy as np

from NBase import NBase


class NRegressor(NBase):

    def predict(self, anet, inp):
        inout = inp[:]
        for lay in anet:
            inout = lay.evaluate(inout)

        res = inout[0]
        return res

    def _score(self, x, y):
        p = self.predict(self._net, x)
        # scr = (p - y)**2 # mse
        scr = np.log(np.cosh(p - y))    # log
        # scr = np.abs(p - y) # abs
        return scr

    def getYhat(self, nx=False, multi=False):
        X = self._idatas["X"]
        if nx:
            X = self._idatas["nX"]
        yhat = np.array([self.predict(self._best_net, x) for x in X])

        return X, yhat

    def test_me(self):
        nX = self._idatas["nX"]
        nY = self._idatas["nY"]

        nYhat = np.array([self.predict(self._best_net, x) for x in nX])
        y_real = nY
        print nYhat, "=====PRED====== BEST"
        print y_real, "=====REAL====== TEST"
        self.display(self._best_net)

'''

@author: dan
'''

from __future__ import division

from copy import deepcopy

import numpy as np

from Layer import Layer
from NBase import NBase


class NClassifier(NBase):

    def predict(self, anet, inp):
        inout = inp[:]
        for lay in anet:
            inout = lay.evaluate(inout)
            #inout = self.relu(inout)
            #inout = self.sigmoid(inout)
            #inout = np.tanh(inout)

        #res = inout
        res = self.softmax(inout)  # with  log score
        return res

    def _score(self, x, y):
        p = self.predict(self._net, x)
        # pm = np.argmax(p)
        # pequals = np.equal(pm, y)
        # scr = (pequals == False)
        # print "pm: " , pm , "y: " , y , "pe: ", pequals , "scr: " , scr

        scr = -np.log(p[y])

        return scr

    def getYhat(self, nx=False, multi=False):
        X = self._idatas["X"]
        if nx:
            X = self._idatas["nX"]
        p = np.array([self.predict(self._best_net, x) for x in X])
        if multi:
            tp = np.argpartition(-p, 2)  # np.argmax(p, 1)
            #y_labels = tp[:,[0,2]]
            yhat = tp[:, :2]
        else:
            yhat = np.argmax(p, axis=1)

        return X, yhat

    def test_me(self):
        nX = self._idatas["nX"]
        nY = self._idatas["nY"]
        p = np.array([self.predict(self._best_net, x) for x in nX])
        y_real = nY
        #cm = self.confusion_matrix(yhats, y_real)
        #print cm
        print np.argmax(p, axis=1), "=====PRED====== BEST"
        print y_real, "=====REAL====== TEST"
        self.display(self._best_net)

    def softmax(self, x):
        e_x = np.exp((x - np.max(x)))
        out = e_x / e_x.sum()
        return out

    def sigmoid(self, x):
        return .5 * (1 + np.tanh(.5 * x))

    def relu(self, x, alpha=.01):
        return np.maximum(x, np.exp(alpha*x) - 1)

'''

@author: dan
'''
from __future__ import division
import numpy as np
from kernels import Kernels


class Ktsne:

    def __init__(self, X, f_opts={}):

        self._dim = f_opts["n_dims"]
        self._perp = f_opts["perplexity"]
        self._eta = f_opts["eta"]
        self.f_opts = f_opts
        self._momentum = .4
        self.X = X.copy()
        self.P = None
        self.Q = None
        self._iter = 0
        (n, d) = self.X.shape
        y_sz = (n, self._dim)
        mw, Mw = (-np.sqrt(6. / sum(y_sz)), np.sqrt(6. / sum(y_sz)))
        # self.Y = np.random.uniform(mw, Mw, y_sz)
        self.Y = np.random.randn(n, self._dim)*Mw
        self.dY = None  # np.zeros((n, self._dim))
        self.Y_t = np.array(
            [np.zeros_like(self.Y), np.zeros_like(self.Y)])
        np.seterr(divide='ignore', invalid='ignore')

    def reduce_X(self):
        XX = self.X.copy()
        p_dims = self.f_opts["p_dims"]
        gamma = self.f_opts["gamma"]
        degree = self.f_opts["p_degree"]
        kernel = self.f_opts["ker"]
        kn = Kernels(XX, k_opts={
                     "kernel": kernel, "gamma": gamma, "degree": degree, "p_dims": p_dims})

        self.X = kn.process_data()

    def L1(self, X):
        D = np.sum(np.abs(X[:, None] - X[None, :], -1))
        return D

    def L2(self, X):
        D = np.sum((X[:, None] - X[None, :])**2, -1)
        return D

    def bin_search(self, d_row, target, tol=1e-2, niter=1000, low=1e-10, up=1e3):

        for i in xrange(niter):
            estimated = (low + up)/2.
            # print estimated
            p_row = np.exp(- (d_row) / estimated)
            sumP = np.sum(p_row)
            p_row = p_row/sumP
            val = self.entropy(p_row)
            dif = np.abs(val-target)
            if dif <= tol:
                break
            if val > target:
                up = estimated
            else:
                low = estimated

        return p_row, estimated

    def entropy(self, p_row):
        H = -np.sum(p_row*np.log2(p_row))
        return H

    def compute_P(self):

        n, d = self.X.shape
        P = np.zeros((n, n))
        sigmas = np.ones((n, 1))
        D = self.L2(self.X)
        target_entropy = np.log2(self._perp)
        for i in xrange(n):
            d_row = D[i, np.hstack((np.arange(0, i), np.arange(i+1, n)))]
            p_row, estimated = self.bin_search(d_row, target_entropy)
            P[i, np.hstack((np.arange(0, i), np.arange(i+1, n)))] = p_row
            sigmas[i] = estimated

        msig = np.mean(np.sqrt(1/sigmas))
        print "Mean value of sigma: ", msig

        # (pik + pki ) the average divide by 2n usualy.. in docs
        P = (P + np.transpose(P)) / (2*n)
        P = P / np.sum(P)
        P = np.maximum(P, 1e-12)
        return P

    def compute_Q(self):
        D = self.L2(self.Y)
        q = 1 / (1 + D)
        np.fill_diagonal(q, 0)
        self.Q = q / np.sum(q)
        self.Q = np.maximum(self.Q, 1e-12)
        return q

    def gradient(self, q):
        PQ = self.P - self.Q
        M = PQ * q
        MD = 4 * (np.diag(np.sum(M, 1)) - M)
        self.dY = np.dot(MD, self.Y)

    def get_solution(self, steps=500):

        self.reduce_X()
        self.P = self.compute_P()
        self.P = self.P * 10.

        for i in xrange(steps):
            cost = self.step()
            if i % 500 == 0:
                print "Iteration ", i, ": cost is ", cost

        print "cost : ", cost
        self.Y = self.Y - np.mean(self.Y, 0)
        #print self.Y[:5]
        return self.Y

    def step(self):

        q = self.compute_Q()
        self.gradient(q)

        if self._iter == 100:
            self.P = self.P / 10.
        if self._iter == 25:
            self._momentum = .8

        self.Y = self.Y - self._eta*self.dY
        self.Y = self.Y + self._momentum * \
            np.diff(self.Y_t, axis=0)[0]
        self.Y_t[1] = self.Y_t[0].copy()
        self.Y_t[0] = self.Y

        C = np.sum(self.P * np.log(self.P / self.Q))
        self._iter += 1

        return C

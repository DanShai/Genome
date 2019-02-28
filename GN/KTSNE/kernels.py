'''

@author: dan
'''
from __future__ import division
import numpy as np


class Kernels:
    def __init__(self, XX, k_opts={"kernel": "pca", "gamma": .5, "degree": 1, "pcomp": 4}):
        self.X = XX.copy()
        self.k_opts = k_opts

    def process_data(self):
        gamma = self.k_opts["gamma"]
        p_degree = self.k_opts["degree"]
        p_comp = self.k_opts["p_dims"]
        ker = self.k_opts["kernel"]
        XX = self.X

        if ker == "poly":
            X = self.poly(XX, gamma=gamma, degree=p_degree,
                          n_components=p_comp).real
        elif ker == "anova":
            X = self.anova(XX, gamma=gamma, degree=p_degree,
                           n_components=p_comp).real
        elif ker == "rbf":
            X = self.rbf(XX, gamma=gamma,  n_components=p_comp).real
        elif ker == "cosine":
            X = self.cosine(XX, n_components=p_comp).real
        elif ker == "iquad":
            X = self.iquad(XX, gamma=gamma, degree=p_degree,
                           n_components=p_comp).real
        elif ker == "cauchy":
            X = self.cauchy(XX, gamma=gamma, n_components=p_comp).real
        elif ker == "fourier":
            X = self.fourier(XX, gamma=gamma, n_components=p_comp).real
        else:
            X = self.pca(XX, n_components=p_comp).real

        return X

    def eignes(self, M, n_components=4):
        (vals, V) = np.linalg.eig(M)
        idx = vals.argsort()[::-1]
        vals = vals[idx].real
        print "---------------- vals: ----------------"
        print n_components, vals.shape
        print vals[:n_components]
        print '---------------------------------------'
        V = V[:, idx]
        U = V[:, 0:n_components]
        return U

    def centerK(self, K):
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        return K

    def pca(self, X, n_components=2):

        (n, d) = X.shape
        X -= np.mean(X, 0)
        _cov = np.cov(X.T)
        U = self.eignes(_cov, n_components=n_components)
        XX = np.dot(X, U)
        return XX

    def poly(self, X, gamma=1, degree=2, n_components=2):

        X -= np.mean(X, 0)
        K = (gamma*X.dot(X.T)+1)**degree

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def rbf(self, X, gamma=.1, n_components=2):
        X -= np.mean(X, 0)
        mat_sq_dists = np.sum((X[None, :] - X[:, None])**2, -1)
        K = np.exp(-gamma*mat_sq_dists)

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def cosine(self, X, n_components=2):
        X -= np.mean(X, 0)
        X_1 = ((X ** 2).sum(axis=1)).reshape(X.shape[0], 1)
        K = X.dot(X.T) / (X_1)

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def iquad(self, X, gamma=1, degree=1, n_components=2):

        X -= np.mean(X, 0)

        # degree = 1.  # better small degree

        dists_sq = np.sum((X[None, :] - X[:, None])**2, -1)
        K = 1. / (dists_sq + gamma**2)**degree

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def cauchy(self, X, gamma=.2, n_components=2):

        X -= np.mean(X, 0)
        dists_sq = np.sum((X[None, :] - X[:, None])**2, -1)
        K = 1 / (1 + dists_sq*gamma)

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def anova(self, X, gamma=.01, degree=1, n_components=2):

        X -= np.mean(X, 0)
        K = np.zeros((X.shape[0], X.shape[0]))
        for d in range(X.shape[1]):
            X_d = X[:, d].reshape(-1, 1)
            K += np.exp(-gamma * (X_d - X_d.T)**2) ** degree

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

    def fourier(self, X, gamma=.1, n_components=2):

        X -= np.mean(X, 0)
        K = np.ones((X.shape[0], X.shape[0]))
        gamma = min(.1, gamma)
        for d in range(X.shape[1]):
            X_1 = X[:, d].reshape(-1, 1)
            K *= (1-gamma ** 2) / \
                (2*(1 - 2*gamma * np.cos(X_1 - X_1.T)) + gamma**2)

        K = self.centerK(K)
        return self.eignes(K, n_components=n_components)

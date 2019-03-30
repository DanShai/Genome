
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from GN.NN.NClassifier import NClassifier
from GN.NN.NRegressor import NRegressor
from GN.Plot.Plotter import Plotter
from GN.Threads.PThread import PThread
from GN.Threads.SetInterval import SetInterval

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NetMain(object):
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.net = None
        self.plotter = None
        self.interval = None
        self.datas = None

    def getXY(self, dt="iris"):
        if dt == "iris":
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
        else:
            digits = datasets.load_digits()
            X = digits.data
            y = digits.target
        return X, y

    def ktsne(self, x, kernel):
        f_opts = {'p_degree': 4.0, 'p_dims': 24, 'eta': 50.0,
                  'perplexity': 20.0, 'n_dims': 2, 'ker': kernel, 'gamma': 0.1}
        k_tsne = Ktsne(x, f_opts=f_opts)
        X_reduced = k_tsne.get_solution(3000)
        X_reduced = self.scaler.fit_transform(X_reduced)
        return X_reduced

    def top_features(self, X, top=2):
        ix = self.net.get_indexes(top)
        print "=====Top features======"
        print ix
        row_ix = ix[0, :]
        X2 = X[:, row_ix]
        t = int(top/2)
        X2_1 = X2[:, :t]
        X2_2 = X2[:, t:]
        X3 = np.hstack((X2_1.mean(1).reshape((-1, 1)),
                        X2_2.mean(1).reshape((-1, 1))))
        return X3

    def plotme(self, *largs):
        gon = self.net.getLoading()
        if not gon:
            self.interval.stop()
            _, Yh = self.net.getYhat(True, False)
            print Yh, "=====PRED======"
            print self.datas["nY"], "=====REAL======"
            X, Yh = self.net.getYhat(False, False)
            Y = self.datas["Y"]
            if self.net.getMode() == "CLA" and X.shape[1] > 2:
                X3 = self.top_features(X, top=2)
                self.plotter.plot(X3, Y, Yh)
            else:
                self.plotter.plot(X, Y, Yh)
            self.plotter.ioff()
        else:
            X, Yh = self.net.getYhat(False, False)
            Y = self.datas["Y"]
            if self.net.getMode() == "CLA" and X.shape[1] > 2:
                X3 = self.top_features(X, top=2)
                self.plotter.plot(X3, Y, Yh)
            else:
                self.plotter.plot(X, Y, Yh)

    def classify(self):

        nrOpts = {"opx": 1, "depth": 1, "nvars": None, "pvc": .6,
                  "pf": 1., "cross": .4, "mut": .5, "mrand": .5}
        gOpts = {"mxepoch": 1500, "bsize": 192, "bupdate": 10, "fraction": .25,
                 "history": 5, "mxtries": 5, "mode": "CLA"}

        print nrOpts
        print gOpts
        X, y = self.getXY("iris")
        #X, y = self.getXY("digits")

        # print i, o, "dim"
        X, y = shuffle(X, y)
        o = np.unique(y).size

        X = X[:500]
        y = y[:500]

        X1 = self.scaler.fit_transform(X)
        kernel = 'pca'
        # X_pca = PCA(n_components=2).fit_transform(X1)
        #XX = X1
        XX = X_pca
        Xtr = XX[:-10]
        Xts = XX[-10:]
        ytr = y[:-10]
        yts = y[-10:]

        i = Xtr.shape[1]
        self.datas = {"X": Xtr, "Y": ytr, "nX": Xts, "nY": yts}

        l_dims = [i,  o]
        self.net = NClassifier(dim=l_dims, datas=self.datas,
                               nr_opts=nrOpts, g_opts=gOpts)
        self.net.setLoading(True)
        self.plotter = Plotter(mode="CLA")
        lf = PThread(target=self.net.train)
        lf.start()
        self.interval = SetInterval(5, self.plotme)

    def reg(self):

        nrOpts = {"opx": 3, "depth": 1, "nvars": None, "pvc": .6,
                  "pf": 1., "cross": .3, "mut": .5, "mrand": .5}
        gOpts = {"mxepoch": 2500, "bsize": 192, "bupdate": 10, "fraction": .25,
                 "history": 5, "mxtries": 5, "mode": "REG"}
        print nrOpts
        print gOpts

        # X = np.random.normal(size=100)
        # XX=np.arange(0, 105, 1)
        # XX = np.random.randint(low=-50, high=50, size=1000)
        XX = np.linspace(-10., 11., num=100)
        YY = (XX - 2) * np.cos(2 * XX)
        # YY = XX**2 + XX - 1
        # Make sure that it X is 2D
        # N = 1000
        # s = 10
        # XX = s*np.random.rand(N)
        # XX = np.sort(XX)
        # YY = np.sin(XX) + 0.1*np.random.randn(N)
        Y = YY[:-5]
        nY = YY[-5:]
        X = XX[:-5]
        X = X[:, np.newaxis]
        nX = XX[-5:]
        nX = nX[:, np.newaxis]
        self.datas = {"X": X, "Y": Y, "nX": nX, "nY": nY}
        i = X.shape[1]
        l_dims = [i, 1]

        self.net = NRegressor(dim=l_dims, datas=self.datas,
                              nr_opts=nrOpts, g_opts=gOpts)
        self.net.setLoading(True)
        self.plotter = Plotter(mode="REG")
        lf = PThread(target=self.net.train)
        lf.start()
        self.interval = SetInterval(5, self.plotme)


if __name__ == "__main__":
    m = NetMain()
    # m.reg()
    m.classify()

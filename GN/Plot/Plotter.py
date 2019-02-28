'''

@author: dan
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, mode="CLA"):
        self._mode = mode
        plt.ion()
        plt.show()

    def ioff(self):
        plt.ioff()
        plt.show()

    def plot(self, X, Y, Yh):
        self.plot_cla(
            X, Y, Yh) if self._mode == "CLA" else self.plot_reg(X, Y, Yh)

    def plot_one(self, X, y, title=" scatter "):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    def plot_cla(self, X, Y, yh):
        plt.clf()
        plt.subplot(2, 1, 1)
        self.plot_one(X, Y, ' real ')
        plt.subplot(2, 1, 2)
        self.plot_one(X, yh, ' predicted ')
        plt.draw()
        plt.pause(0.001)

    def plot_reg(self, X, Y, yh):
        plt.clf()
        plt.plot(Y, color='orange')
        plt.plot(yh, color='c')
        plt.draw()
        plt.pause(0.001)

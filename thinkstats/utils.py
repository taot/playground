import copy
import bisect
import numpy as np
from matplotlib import pyplot as plt

class _DictWrapper(object):

    def __init__(self, obj=None):
        self.d = {}
        if obj is None:
            return
        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, _DictWrapper):
            self.d.update(obj.items())
        else:
            for i in obj:
                if (np.isnan(i)):
                    continue
                i = round(i * 100) / 100.0
                self.d[i] = self.d.get(i, 0) + 1

        if isinstance(self, Pmf):
            self.normalize()

    def items(self):
        return self.d.items()

    def copy(self):
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        return new

    def total(self):
        return sum(self.d.values())

    def sum(self):
        sum = 0.0
        for x, v in self.d.items():
            sum += x * v
        return sum

    def mult(self, x, m):
        if not x in self.d:
            return
        self.d[x] *= m

    def mean(self):
        return self.sum() / self.total()

    def show(self, **options):
        plt.bar(list(self.d.keys()), list(self.d.values()), **options)
        plt.show()

class Hist(_DictWrapper):
    pass

class Pmf(_DictWrapper):

    def normalize(self):
        d2 = {}
        s = sum(self.d.values())
        for x, v in self.items():
            d2[x] = float(v) / s
        self.d = d2

class Cdf(object):

    def __init__(self, obj=None):
        if obj is None:
            return
        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)
            return

        cumsum = 0.0
        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)
        xs, freqs = zip(*sorted(dw.d.items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=np.float32)
        self.ps /= self.ps[-1]

    def prob(self, x):
        index = bisect.bisect(self.xs, x)
        return self.ps[index - 1]

    def value(self, x):
        index = bisect.bisect(self.ps, x)
        return self.xs[index - 1]

    def show(self, **options):
        plt.plot(self.xs, self.ps, **options)
        plt.show()

def histogram(series):
    hist = {}
    for i in series:
        i = int(i)
        hist[i] = hist.get(i, 0) + 1
    hist
    plt.bar(list(hist.keys()), list(hist.values()))
    plt.show()

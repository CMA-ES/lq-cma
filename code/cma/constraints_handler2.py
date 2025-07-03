# -*- coding: utf-8 -*-
"""2019: constraints handling classes based on ranks

TODO: rename to constraints_by_rank
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
del absolute_import, division, print_function  #, unicode_literals
# __package__ = 'cma'

import numpy as np

from . import fitness_models
from .utilities import utils as _utils

def _progressive_ranks(arf):
    """return progressive rank values as int((r**2 + r) / 2) = 0, 1, 3, 6, 10, 15, 21,..."""
    rs = _utils.ranks(arf)
    return [int((n + n**2) / 2) for n in rs]

def _new_rank_correlation(X, ranks=None):
    """compute correlation between ranks and <x_best-x_worst, x>-ranking"""
    if ranks is None:  # doesn't make sense?
        ranks = list(range(len(X)))
        imin, imax = 0, len(X) - 1
    else:
        imin, imax = np.argmin(ranks), np.argmax(ranks)

    xlin = np.asarray(X[imax]) - X[imin]
    flin = [sum(xlin * x) for x in X]
    print(imin, imax)
    print(flin, ranks)
    print(_utils.ranks(flin), _utils.ranks(ranks))
    return fitness_models._kendall_tau(flin, ranks)

def nbfeas(g_values):
    """return number of `g_values` that are <= 0"""
    return sum(g <= 0 for g in g_values)

class Glinear:
    """a linear bound constraint function on variable i.
    
    Calling the constraint gives `sign` times ``x - bound_value``
    """
    def __init__(self, i, sign=1, bound_value=0, concave_coefficient=1):
        self.i = i
        self.sign = sign
        self.bound_value = bound_value
        self.concave_coefficient = concave_coefficient
    def __call__(self, x):
        g = self.sign * (x[self.i] - self.bound_value)
        return g * (1 if g < 0 else self.concave_coefficient)
        
class RankWeight:
    """A weight that adapts depending on observed values.
    
    The weight is designed to be used in a ranksum.

    TODO: should the threshold be a percentile instead of all(.)?
    """
    def __init__(self, increment=None):
        """`increment` default is 1.
        
        An `increment` around 0.9 gives smallest axis ratio on the "sphere".
        """
        self.weight = 1
        self.increment = increment if increment is not None else 1
    def update(self, g_vals):
        """update weight depending on sign values of `g_vals`.

        Increment by a constant if all values are larger than zero,
        reset to one otherwise.
        """
        assert len(g_vals) > 0  # otherwise we have no basis for adaptation
        if all(g > 0 for g in g_vals):
            self.weight += self.increment
        else:
            self.weight = 1
        return self.weight
    
class FSumRanks:
    """Testing rank-based optimization of individual functions,
    for example the sphere function formulated in each coordinate
    as f_i(x) = x_i^2.
    
    We try to optimize the weighted sum of the ranks from each function.

    Outcome: because the partial derivatives do not grow, variables can
    escape for a very long time and hence need larger weights when the are
    further away from the optimum.

    Potential (implemented) solution: the weight reflects the time how long
    the constraint was never satisfied, which correlates with the distance
    and hence with the partical derivative.
    """
    def __init__(self, f_vec):
        """f_vec is a list or tuple of functions or an integer"""
        if not hasattr(f_vec, '__getitem__'):
            assert not callable(f_vec)
            f_vec = [Glinear(i) for i in range(f_vec)]
        self.f_vec = f_vec
        self.weights = [RankWeight() for _ in range(len(self.f_vec))]
        self._xopt = 0

    def distance(self, x):
        return sum((np.asarray(x) - self._xopt)**2)**0.5

    def __call__(self, X):
        ranks = np.zeros(len(X))
        for w_i, f_i in zip(self.weights, self.f_vec):
            fs = [f_i(x) for x in X]
            w_i.update(fs)
            # TODO: the abs value should preferably be part of the function? The "problem" is that we use Glinear as default.
            ranks += w_i.weight * (1 + np.asarray(_utils.ranks([np.abs(f) for f in fs]))**2)
        return list((1 + ranks / 1e2) * self.distance(X[0])**2)

class GSumRanks:
    """Compute a rank-based penalty from constraints functions.

    The penalty for feasible solutions is zero. The penalty for the i-th best
    infeasible solution is in ]i-1, i[.
    """
    def __init__(self, g_vec, increment=None):
        """g_vec is a list or tuple of functions"""
        self.g_vec = g_vec
        self.weights = [RankWeight(increment) for _ in range(len(self.g_vec))]
        self._glimit = 0  # CAVEAT: changing this breaks RankWeight.update

    def distance(self, G, normtype=1):
        """G-space distance to feasibility of g-values vector G"""
        G0 = np.asarray(G) - self._glimit
        G1 = (G0 > 0) * G0
        return sum(G1**normtype)**(1 / normtype)

    @property
    def distances(self):
        """depends on g-values of last call"""
        g_array = np.asarray(self.g_array).T  # now a row per offspring
        return [self.distance(gs) for gs in g_array]

    def g_transform(self, g):
        """maps g in [0, inf] to [0, 1/2]"""
        return np.tanh(g) / 2.  # np.exp(x) / (1 + np.exp(x)) - 0.5

    def __call__(self, X):
        """return weighted ranksum plus tanh(sum g+) / 2 for each x in X.
        
        TODO: The penalty of the best infeasible solution is in ]0,1[.
        This may not be desired and somewhat counteracts the weight adaptation.
        Which weight adaptation? 
        """
        ranks = np.zeros(len(X))
        self.g_array = []
        if not len(self.g_vec):
            return ranks
        for w_i, g_i in zip(self.weights, self.g_vec):
            gs = [g_i(x) for x in X]
            self.g_array += [gs]
            w_i.update(gs)  # this is why we need to loop over g first
            rs = np.asarray(_utils.ranks(gs)) + 1 - nbfeas(gs)
            rs[rs < 0] = 0
            ranks += w_i.weight * rs
        # TODO/check: with - min(ranks) the penalty of the best infeasible is < 1
        return list(ranks - 1 * min(ranks) + self.g_transform(self.distances))
       
class FGSumRanks:
    """TODO: prevent negative f-values but how?"""
    def __init__(self, f, g_vec, increment=None):
        self.f = f
        self.granking = GSumRanks(g_vec)
        self.weight = 1
        self.increment = increment if increment is not None else 1

    def __call__(self, X):
        """return weighted rank plus small offset"""
        fs = [self.f(x) for x in X]
        if not len(self.granking.g_vec):
            return fs
        g_ranks = self.granking(X)
        self.update(g_ranks)
        fg_offset = -max((min(fs), min(self.granking.distances)))
        # TODO: seems wrong, offset should be something like min(fs + gdistance) and ideally not smaller than a feasible f-value?
        # that is: either min feasible f or max infeasible f? 
        return [fg_offset + self.weight * f_rank + g_rank
                for (f_rank, g_rank) in zip(_utils.ranks(fs), g_ranks)]

    def threshold(self):
        # TODO: rename to constraints_OK
        # TODO: use 2%itle and reconsider the max weight threshold, max should be a percentile?
        raise NotImplementedError("need to be read and understood")
        return max(w.weight for w in self.granking.weights) < 2

    def update(self, g_ranks):
        """update weight depending on feasibility ratios"""
        g_array = np.asarray(self.granking.g_array)
        if np.sum(g_array > 0) < g_array.size / 3 and self.threshold():
            self.weight += self.increment
        elif np.sum(g_array > 0) > g_array.size * 2 / 3:
            self.weight -= self.increment
        self.weight = min((max(g_ranks)), self.weight)  # set upper bound, TODO: must depend on # of g?
        self.weight = max((1, self.weight))  # lower bound is 1


class SignTracker:
    """count the number of times g changes to the more extreme
    and reset when crossing zero.
    """
    def __init__(self):
        self.counts = None
    def _init_(self, g):
        self.g = g
        self.counts = np.array(len(g) * [0.])
    def __call__(self, g):
        """track g, dg, count(previous_g * dg > 0)"""
        self.counts is None and self._init_(g)
        assert np.all(self.counts * self.g >= 0)
        self.dg = np.asarray(g) - self.g
        idx = self.dg * self.g > 0  # previous g and dg have the same sign
        self.counts[idx] += np.sign(self.dg[idx])  # .astype(int) seems to be slower
        self.counts[self.counts * g <= 0] = 0
        self.g = g
        return self
    def trigger(self, n):
        """Not in use.
        
        Trigger with gaps of n, 1 + sqrt(n), 1 + sqrt(1 + sqrt(n)), 1 + sqrt(1 + sqrt(1 + sqrt(n)))...
        and limit 2.618033988749895.

        Alternative: gaps of n, sqrt(n + 1),... with limit 1.618033988749895"""
        c = np.abs(self.counts)
        t = 0 * c
        while np.any(c > 0):
            c -= int(n)
            t += c == 0
            n = n**0.5 + 1
        res = (t > 0) * (self.counts * self.dg > 0)
        return res


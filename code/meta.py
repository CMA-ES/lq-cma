#!/usr/bin/env python
"""Optimize meta parameters of LQ-CMA use UH-CMA.

Current 3 parameters.

"""
from __future__ import division, print_function, unicode_literals
del division, print_function, unicode_literals

import numpy as np
import cma
import cocoex

class BBOBFunc:
    def __init__(self, fid, dimension, instance=1):
        suite = cocoex.Suite('bbob', '', '')
        self.fun = suite.get_problem_by_function_dimension_instance(
            fid, dimension, instance)
        self.fun._best_parameter('print')
        with open("._bbob_problem_best_parameter.txt") as file_:
            self.xopt = np.asarray([float(xi) for xi in file_.read().split()])
        self.fopt = self.fun(self.xopt)
    def __call__(self, x):
        return self.fun(self.xopt + x) - self.fopt

class Parallelizer:
    def __init__(self, f):
        self.f = f
    def __call__(self, X):
        if isinstance(X, list) and not np.isscalar(X[0]):
            return [self.f(x) for x in X]
        return self.f(X)

class MetaFitness:
    """determine fitness of parameter setting for lq-cma by running it"""
    def __init__(self, function, dimension):
        self.x0 = dimension * [0.1]
        self.sigma0 = 0.1
        self.ftarget = 1e-7
        self.maxevals = 200 * dimension
        self.function = function
        self.dimension = dimension

    def __call__(self, theta):
        """theta are the meta parameters, currently 2"""
        return self.eval(theta)

    def eval(self, theta):
        if theta[0] < theta[1]:
            return self.pen_fitness(theta)

        import cma.fitness_models as fm
        fm.Logger = fm.LoggerDummy  # do not log
        surrogate = fm.SurrogatePopulation(self.function,
                                           fm.LQModel(theta[0], theta[1]),
                                           model_size_factor=3)
        inject_xopt = fm.ModelInjectionCallback(surrogate.model)

        xopt, es = cma.fmin2(self.function, self.x0, self.sigma0,
                        {'maxfevals': self.maxevals,
                         'ftarget': self.ftarget,
                         'CMA_injections_threshold_keep_len': 1,
                         'verbose': -9},
                        restarts=7,
                        parallel_objective=surrogate,
                        callback=[inject_xopt,],
                    )
        return self.meta_fitness(es)

    def pen_fitness(self, theta):
        return 1e6 + 1e4 * (theta[1] - theta[0])

    def meta_fitness(self, es):
        res = es.result
        val = ('ftarget' not in es.stop()) * (self.maxevals + max((0, res.fbest)))
        val += res.evaluations
        return val

def get_es(esin):
    global es
    es = esin

class ParameterTransformatorBase:
    @property
    def dimension(self):
        return len(self.x0)
    def inverse(self, y):
        """

        >>> from meta import ParameterTransformator
        >>> p = ParameterTransformator()
        >>> x = [0.1, 0.2, 0.3]
        >>> assert np.allclose(x, p(p.inverse(x)))

        """
        raise NotImplementedError
    def __call__(self, x):
        """transfrom from optimizer to fitness space"""
        return np.asarray(self.transfrom(x))

class ParameterTransformator(ParameterTransformatorBase):
    """totally hand-crafted improvised class, also defineds x0"""
    def __init__(self):
        self.x0 = self.inverse([1, 0.5, 0.85])
        self.sigma0 = 0.1

    def transfrom(self, x):
        return [x[0]**2, x[1]**2, 1 - (1 - x[2])**2]

    def inverse(self, y):
        return [y[0]**0.5, y[1]**0.5, -(np.sqrt(-(y[2] - 1)) - 1)]

dimension = 10
function = BBOBFunc(14, dimension) # or 7step (caveat restarts)
# function = cma.ff.sectorsphere
# function = cma.ff.rosen

if __name__ == '__main__':
    fitness = MetaFitness(function, dimension)
    parameters = ParameterTransformator()
    x, es = cma.fmin2(Parallelizer(cma.fitness_transformations.ComposedFunction(
                                [fitness, parameters])),
                      parameters.x0, parameters.sigma0,
        {
            # 'verb_filenameprefix': 'outcma3/',
        },
                      noise_handler=cma.NoiseHandler(dimension, [1, 1, 100], aggregate=np.mean,
                                                     parallel=False),
                    callback=get_es)
    es.result_pretty()
    print(parameters.inverse(es.mean))
"""
Result:
sectorsphere: 1, 0.5
f12bent:
"""
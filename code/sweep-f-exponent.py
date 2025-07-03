#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test invariance to monotone f-transformation"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import collections
import shutil
import tempfile
if sys.platform.lower() not in ('darwin', 'windows'):
    import mkl
    mkl.set_num_threads(1)
import numpy as np
from scipy.optimize import fmin_slsqp, fmin_bfgs
import cma
import cma.experimentation
import cma.fitness_models as fm
import cocoex

class InfolderGoneWithTheWind:
    """``with InfolderGoneWithTheWind(): ...`` executes the block in a

    temporary folder under the current folder. The temporary folder is
    deleted on exiting the block.

    >>> import os
    >>> dir_ = os.getcwd()  # for the record
    >>> len_ = len(os.listdir('.'))
    >>> with InfolderGoneWithTheWind():  # doctest: +SKIP
    ...     # do some work in a folder here, e.g. write files
    ...     len(dir_) > len(os.getcwd()) and os.getcwd() in dir_
    True
    >>> # magically we are back in the original folder
    >>> assert dir_ == os.getcwd()
    >>> assert len(os.listdir('.')) == len_

    """
    def __init__(self, prefix='_'):
        """no folder needs to be given"""
        self.prefix = prefix
    def __enter__(self):
        self.root_dir = os.getcwd()
        # self.target_dir = tempfile.mkdtemp(prefix=self.prefix, dir='.')
        self.target_dir = tempfile.mkdtemp(prefix=self.prefix)
        self._target_dir = self.target_dir
        os.chdir(self.target_dir)
    def __exit__(self, *args):
        os.chdir(self.root_dir)
        if self.target_dir == self._target_dir:
            shutil.rmtree(self.target_dir)
        else:
            raise ValueError("inconsistent temporary folder name %s vs %s"
                             % (self._target_dir, self.target_dir))

class BBOBFunc:
    def __init__(self, fid, dimension, instance=1):
        suite = cocoex.Suite('bbob', '', '')
        self.fun = suite.get_problem_by_function_dimension_instance(
            fid, dimension, instance)
        with InfolderGoneWithTheWind():
            self.fun._best_parameter('print')
            with open("._bbob_problem_best_parameter.txt") as file_:
                self.xopt = np.asarray([float(xi) for xi in file_.read().split()])
        self.fopt = self.fun(self.xopt)
    def __call__(self, x):
        return self.fun(self.xopt + x) - self.fopt

def nruns(dimension):
    return 1 + int(20 / dimension)

names = [#'slsqp',  # '-fsphere-10d'
         #'bfgs',
         #'cma',
         'lqcma',
         ]
fnames = [#'rosen',
                #'sphere',
                #'elli',
                #'ellirot',
                #'bbob02',
                #'bbob06',
                ###'bbob08', 'bbob09',
                #'bbob10',
                #'bbob11',
                ###'bbob12',
                ###'bbob13',
                #'bbob14',
                 ]
if __name__ == "__main__" and len(sys.argv) > 1:
    fnames = [sys.argv[1]]

experiment_number = 1  # to create filename
dimensions = [2, 3, 5, 10, 20]
alpha_range = np.logspace(0, np.log10(16), 7)  # np.logspace(0, 1.2, 9)
alpha_ranges = [alpha_range, [1. / a for a in alpha_range[1:]]]

if __name__ == "__main__":
    for dimension in dimensions:
        maxfevals = 100 + dimension ** 1.5 * 2000
        # maxfevals = 100 + 2 * dimension
        number_of_restarts = 20
        for fname in fnames:
            for name in names:
                filename = '-'.join([name, fname, '%02d' % dimension, '%d' % experiment_number])
                evals = cma.experimentation.Results(filename)
                for irun in range(nruns(dimension)):
                    for alphas in alpha_ranges:
                        last_alpha = 0
                        for alpha in alphas:
                            if np.nan in evals.data[alpha]:
                                break
                            if last_alpha in evals.data and np.nan in evals.data[last_alpha]:
                                break
                            last_alpha = alpha
                            try:
                                fraw = getattr(cma.ff, fname)
                            except:
                                fraw = BBOBFunc(int(fname.split('bbob')[1]), dimension)
                            fun = cma.fitness_transformations.Function(
                                    lambda x: fraw(x)**alpha)
                            fun.ftarget = 1e-1**alpha
                            x0 = -2 + 0.1 * np.random.randn(dimension)
                            if name == 'slsqp':
                                for i in range(number_of_restarts):
                                    while not fun.target_hit_at and fun.evaluations < maxfevals:
                                        output = fmin_slsqp(fun, x0,
                                                            acc=1e-11, iter=10000,
                                                            full_output=True, iprint=-1)
                            elif name == 'cma':
                                output = cma.fmin2(fun, x0, 1,
                                                   {'ftarget': fun.ftarget, 'maxfevals':maxfevals,
                                                    'conditioncov_alleviate': [np.inf, np.inf],  # DO NOT REMOVE THIS
                                                    'verbose':-9},
                                                  restarts=9)
                            elif name == 'lqcma':
                                surrogate = fm.SurrogatePopulation(fun)
                                inject = fm.ModelInjectionCallback(surrogate.model)
                                output = cma.fmin2(None, x0, 1,
                                                   {'ftarget': fun.ftarget, 'maxfevals':maxfevals,
                                                    # 'CMA_injections_threshold_keep_len': 1,
                                                    'conditioncov_alleviate': [np.inf, np.inf],  # DO NOT REMOVE THIS
                                                    'verbose':-9},
                                                    parallel_objective=surrogate,
                                                    callback=inject,
                                                    restarts=7)
                            elif name == 'bfgs':
                                for i in range(number_of_restarts):
                                    while not fun.target_hit_at and fun.evaluations < maxfevals:
                                        output = fmin_bfgs(fun, x0,
                                                           maxiter=10000,
                                                           gtol=1e-11,  # default is 1e-5, makes a difference only for alpha>1?
                                                           full_output=True, disp=False)

                            else:
                                raise NotImplementedError
                            if fun.target_hit_at:
                                evals.data[alpha] += [fun.target_hit_at]
                            else:
                                evals.data[alpha] += [np.nan]
                                print(alpha, name, fname, output)
                                try: print(output[1].stop(), output[1].result)
                                except: pass
                                evals.save()
                                break
                            evals.save()

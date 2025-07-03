#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher of any number of batches.

Example to lauch 8 batches::

    python launch.py example_experiment2.py budget_multiplier=3 8

executes::

    nohup nice python -u example_experiment2.py budget_multiplier=3 batch=i/8

for ``i in range(8)``.

After that::

    ps ax -o  uid,pid,ni,%mem,%cpu,time,etime,pid,cmd | sort -n  | tail -n 55

show the number of running processes.

"""
from __future__ import division, print_function, unicode_literals
del division, print_function, unicode_literals

import os, sys

def cmd(s):
    print(s)
    os.system(s)  # comment for dry run

if __name__ == "__main__":
    python_name = 'python'
    if 11 < 3:
        python_name = './python' + os.getcwd()[-3:]
        cmd("cp -p %s %s" % (sys.executable, python_name))  # only to show different names in top
    script_name = "../%s/%s" % (os.path.split(os.getcwd())[-1], sys.argv[1])

    if sys.argv[1].startswith(('meta.py', )):
        cmd("nohup nice %s -u %s > out.txt 2> err.txt &" % (python_name, script_name))
    elif sys.argv[1].startswith('sweep-f-exponent.py'):
        if  len(sys.argv) > 2:
            funnames = [sys.argv[2]]
        else:
            funnames = [
                'sphere',
                'elli',
                'ellirot',
                'bbob02',
                'bbob06',
                'bbob10',
                'bbob11',
                'bbob14',
            ]
        for funname in funnames:
            cmd("nohup nice %s -u %s %s > out-%s.txt 2> err-%s.txt &" % (
                python_name, script_name, funname, funname, funname))
    elif sys.argv[1].startswith('example_experiment2.py'):
        if len(sys.argv) < 3:
            raise ValueError("need at least python file and number of batches as argument")
        batches = int(sys.argv[-1])

        for i in range(batches):
            cmd(("nohup nice %s -u %s " % (python_name, script_name)) +
                ' '.join(sys.argv[2:-1]) +
                ' batch=%d/%d > out.txt%d 2> err.txt%d &' % (i, batches, i, i)
                )
    else:
        print("command file %s not found" % sys.argv[1])


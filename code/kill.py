#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kill range of processes, like::

    ./kill.py 1234 1236

kills all processes from 1234 to 1236 included. Asks for explicite confirmation.

"""
from __future__ import division, print_function, unicode_literals
del division, print_function, unicode_literals
import os, sys

if sys.version[0] == '2':  # in python 2
    input = raw_input  # in py2, input(x) == eval(raw_input(x))

def cmd(s):
    print(s)
    os.system(s)  # comment for dry run

if __name__ == "__main__":
    from_, to_ = sys.argv[1], sys.argv[2]
    rg = list(range(int(from_), int(to_) + 1))
    print("Killing up to %d processes from %s to %s, type 'y' + return to proceed" % (len(rg), from_, to_))
    response = input()
    if response == 'y':
        for i in rg:
            cmd('kill %d' % i)
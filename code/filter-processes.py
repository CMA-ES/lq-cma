#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minitool to filter output of ps by CPU%

First argument is line length if > 100 and percentage threshold if < 100.
"""
import os, sys, time

maxlinewidth = 160
percentage_threshold = 1.1  # more than 1.1% CPU time
try:
    arg1 = float(sys.argv[1])
except: pass
else:
    if arg1 < 100:
        percentage_threshold = arg1
    else:
        maxlinewidth = int(arg1)

def cmd(s):
    print(s)
    os.system(s)  # comment for dry run

print("""  to filter, type: "ps ax -o uid,uname,pid,ni,%mem,%cpu,time,etime,pid,cmd | sort -n | filter-processes.py" """)
i = 0
while True:
    line = sys.stdin.readline()[:-1]  # remove line break
    try:
        if float(line.split()[5]) > percentage_threshold:
            i += 1
            print('%2d' % i, line[:maxlinewidth])
    except (ValueError, IndexError):  # float conversion didn't work
        print(line[:maxlinewidth])

    if not line:  # nothing more to read
        break

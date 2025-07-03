ps ax -o  uid,uname,pid,ni,%mem,%cpu,time,etime,pid,cmd | sort -n  | python filter-processes.py | less


#coding: utf-8
import datetime
import time

from TextMain import Main

start_time0 = datetime.time(22, 0, 0)
end_time0 = datetime.time(23, 59, 59)
start_time1 = datetime.time(0, 0, 0)
end_time1 = datetime.time(6, 0, 0)
##　----------------------------------------------------------------------
sleep_seconds = 30*60
count, max_count = 0, 100
while 1:
    date_time = datetime.datetime.now()
    time_ = date_time.time()
    if start_time0<=time_<=end_time0 or start_time1<=time_<=end_time1:
        print "Execution time:", time_
        Main()
        count += 1
    else:
        print "Waiting time:", time_
    if count == max_count:
        break
    time.sleep(sleep_seconds)

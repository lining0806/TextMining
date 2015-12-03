#coding: utf-8
import datetime
import time

from TextMain import Main

start_time = datetime.time(0, 0, 0)
end_time = datetime.time(6, 0, 0)
# print start_time<end_time
##　----------------------------------------------------------------------
sleep_seconds = 30*60
count, max_count = 0, 100
while 1:
    date_time = datetime.datetime.now()
    time_ = date_time.time()
    if start_time<time_<end_time:
        print time_
        Main()
        count += 1
    if count == max_count:
        break
    time.sleep(sleep_seconds)

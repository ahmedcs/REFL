import os
import sys

job_name = sys.argv[1]
time_stamp = sys.argv[2]

if job_name == 'all':
    os.system("ps -ef | grep python | grep REFL > refl_running_temp")
elif time_stamp != '':
    os.system("ps -ef | grep python | grep job_name={} | grep time_stamp={} > refl_running_temp".format(job_name, time_stamp))
else:
    os.system("ps -ef | grep python | grep job_name={} > refl_running_temp".format(job_name))

lines = open("refl_running_temp").readlines()
print(lines)
[os.system("kill -9 "+str(l.split()[1]) + " 1>/dev/null 2>&1") for l in lines]
os.system("rm refl_running_temp")

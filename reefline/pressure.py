"""
pressure.py
"""
from datetime import datetime, timedelta
import numpy as np
import os


def read_pressure_from_toa5(filenames):
    """Reads pressure data from TOA5 file written by
    the Campbell Scientific logger. If filenames is a string, 
    process a single file. If it is a list of strings, 
    process files in order and concatenate."""
    if type(filenames) is str:
        print('Reading ', filenames)
        data = [line.rstrip() for line in open(filenames).readlines()[4:]]
    elif type(filenames) is list:
        data = []
        for filename in filenames:
            print('Reading ', os.path.basename(filename))
            data += [line.rstrip() for line in open(filename).readlines()[4:]]
    else:
        raise RuntimeError('filenames must be string or list')

    p1, p2, p3, p4, p5, p6, p7, p8, times = [], [], [], [], [], [], [], [], []
    for line in data:
        line = line.replace('"', '').split(',')
        t = line[0]
        if len(t) == 19:
            time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        elif len(t) == 21:
            time = datetime.strptime(t[:19], '%Y-%m-%d %H:%M:%S')
            time += timedelta(seconds=float(t[-2:]))
        else:
            time = datetime.strptime(t[:19], '%Y-%m-%d %H:%M:%S')
            time += timedelta(seconds=float(t[-3:]))
        times.append(time)
        p1.append(float(line[2]))
        p2.append(float(line[3]))
        p3.append(float(line[4]))
        p4.append(float(line[5]))
        p5.append(float(line[6]))
        p6.append(float(line[7]))
        p7.append(float(line[8]))
        p8.append(float(line[9]))
    return np.array(times), np.array(p1), np.array(p2), np.array(p3),\
        np.array(p4), np.array(p5), np.array(p6), np.array(p7), np.array(p8)

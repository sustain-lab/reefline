from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import numpy as np
from reefline.dispersion import w2k
from reefline.pressure import read_pressure_from_toa5
from reefline.udm import read_udm_from_toa5
from reefline.utility import power_spectrum, running_mean, write_to_csv
from scipy.signal import detrend
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

aquadopp1_path = '/home/milan/Work/sustain/reefline/data/Aquadopp/BASL705'
aquadopp2_path = '/home/milan/Work/sustain/reefline/data/Aquadopp/BASL605'

def read_aquadopp_time(path: str):
    data = [line.strip() for line in open(path).readlines()]
    num_records = len(data)
    time = []
    for line in data:
        line = line.split()
        month, day, year, hour, minute = [int(x) for x in line[:5]]
        seconds = float(line[5])
        ms = int((seconds - int(seconds)) * 1e6)
        time.append(datetime(year, month, day, hour, minute, int(seconds), ms))
    return time

def read_aquadopp_velocity(path: str):
    data = [line.strip() for line in open(path).readlines()]
    num_records = len(data)
    num_cells = len(data[0].split()) - 2
    vel = np.zeros((num_records, num_cells))
    for n in range(num_records):
        vel[n,:] = np.array([float(x) for x in data[n].split()[2:]])
    return vel

time1 = read_aquadopp_time(aquadopp1_path + '.sen')
time2 = read_aquadopp_time(aquadopp2_path + '.sen')

u1 = read_aquadopp_velocity(aquadopp1_path + '.v1')
v1 = read_aquadopp_velocity(aquadopp1_path + '.v2')
w1 = read_aquadopp_velocity(aquadopp1_path + '.v3')

u2 = read_aquadopp_velocity(aquadopp2_path + '.v1')
v2 = read_aquadopp_velocity(aquadopp2_path + '.v2')
w2 = read_aquadopp_velocity(aquadopp2_path + '.v3')



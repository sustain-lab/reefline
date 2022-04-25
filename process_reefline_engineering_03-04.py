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

def title(height, period, angle):
    return 'Hs = %i ft, Tp = %i s, %s' % (height, period, angle)

def filename(height, period, angle, run):
    return ('%ift_%is_%s_%s' % (height, period, angle, run)).replace(' ', '_')

def despike(x, xmax=1.55, xmin=1.25):
    y = np.zeros((x.size))
    y[:] = x[:]
    for n in range(1, x.size - 1, 1):
        if x[n] > xmax:
            y[n] = 0.5 * (x[n-1] + x[n+1])
        if x[n] < xmin:
            y[n] = 0.5 * (x[n-1] + x[n+1])
    return y

matplotlib.rcParams.update({'font.size': 14})

f = 0.38
depth = 0.44
omega = 2 * np.pi * f
#k = np.array([w2k(omega[n], h=depth[n])[0][0] for n in range(f.size)])
k = w2k(omega, h=depth)[0][0]
Cp = omega / k

height = [4, 4, 4, 8, 8, 8, 12, 12, 12, 12, 12, 12]
run = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
period = 14
angle = 9 * ['orthogonal'] + 3 * ['45 deg. oblique']

# all 30-second runs
runs_start = [

    datetime(2022, 3, 4, 19,  5, 12), # a = 0.03, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19,  9, 12), # a = 0.03, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19, 13, 12), # a = 0.03, f = 0.38, orthogonal

    datetime(2022, 3, 4, 19, 17, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19, 21, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19, 25, 12), # a = 0.06, f = 0.38, orthogonal

    datetime(2022, 3, 4, 19, 29, 11), # a = 0.09, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19, 33, 11), # a = 0.09, f = 0.38, orthogonal
    datetime(2022, 3, 4, 19, 37, 14), # a = 0.09, f = 0.38, orthogonal

    #datetime(2022, 3, 4, 19, 41, 11), # a = 0.12, f = 0.38, orthogonal

    datetime(2022, 3, 4, 19, 52, 11), # a = 0.09, f = 0.38, 45 degree oblique
    datetime(2022, 3, 4, 19, 56, 13), # a = 0.09, f = 0.38, 45 degree oblique
    datetime(2022, 3, 4, 19, 52, 11), # a = 0.09, f = 0.38, 45 degree oblique

]

run_seconds = 30

# Elevation from UDM
udm_files = glob.glob('data/udm/TOA5_SUSTAIN_ELEVx6d*2022_03_04*')
udm_files.sort()
time, u1, u2, u3, u4, u5, u6 = read_udm_from_toa5(udm_files)

for n in range(3):
    u1 = despike(u1)
    u6 = despike(u6)

# Pressure
pressure_files = glob.glob('data/pressure/TOA5_SUSTAINpresX4X2.pressure_*_2022_03_04*')
pressure_files.sort()
pressure_time, _, p2, _, p4, p5, _, p7, p8 = read_pressure_from_toa5(pressure_files)

# Pressure set up on 3/4 (full model, 3 rows of cars)
#     Downwave (beachward)
#          4 p8 (Setra)
#   CAR CAR CAR CAR
#          3 p4 (MKS), p5 (Setra), via splitter
#     CAR CAR CAR CAR
#          2 p2 (MKS)
#   CAR CAR CAR CAR
#          1 p7 (Setra)
#     Upwave (inletward)

pressure = np.zeros((4, p2.size))
pressure[0] = p7
pressure[1] = p2
pressure[2] = p4
pressure[3] = p8

num_sensors = 6
num_runs = len(runs_start)
num_records = run_seconds * 20 + 1
num_records_pressure = run_seconds * 10 + 1

e1 = np.zeros((num_runs, num_records))
e2 = np.zeros((num_runs, num_records))
pa = np.zeros((num_runs, num_records_pressure))
pb = np.zeros((num_runs, num_records_pressure))
pc = np.zeros((num_runs, num_records_pressure))
pd = np.zeros((num_runs, num_records_pressure))

for n, t1 in enumerate(runs_start):
    t2 = t1 + timedelta(seconds=run_seconds)

    mask = (time >= t1) & (time <= t2)
    e1[n,:] = detrend(- u1[mask])
    e2[n,:] = detrend(- u6[mask])

    pressure_mask = (pressure_time >= t1) & (pressure_time <= t2)
    pa[n,:] = running_mean(detrend(pressure[0,pressure_mask]), 3)
    pb[n,:] = running_mean(detrend(pressure[1,pressure_mask]), 3)
    pc[n,:] = running_mean(detrend(pressure[2,pressure_mask]), 3)
    pd[n,:] = running_mean(detrend(pressure[3,pressure_mask]), 3)

udm_mask = [0, 2, 4]
x = np.array([-0.63, 0.31, 2.14])
x -= x[1]

dx = x[2] - x[0] # distance between UDM out and UDM in
time_shift = dx / Cp # time shift for UDM out

f = open('reefline_eng.csv', 'w')
f.write('experiment,height,period,measured_height_in,measured_height_out,percent_change\n')

for n in range(num_runs):

    t = np.linspace(0, 30, num_records)
    mask1 = (t >= 5) & (t < 12)
    t1 = t[mask1]

    t2 = np.round(t - time_shift, 2)
    mask2 = (t2 >= 5) & (t2 < 12)
    t2 = t2[mask2]

    e1s = running_mean(e1[n,:], 3)
    e2s = running_mean(e2[n,:], 3)

    w1 = detrend(e1s[mask1])
    w2 = detrend(e2s[mask2])

    hmax1 = np.max(w1) - np.min(w1)
    hmax2 = np.max(w2) - np.min(w2)
    t1 -= t1[0]
    
    tp = np.linspace(0, 30, num_records_pressure)
    pmask = (tp >= 5) & (tp < 12)
    pres1 = pa[n,pmask]
    pres2 = pb[n,pmask]
    pres3 = pc[n,pmask]
    pres4 = pd[n,pmask]
    tp = tp[pmask]
    tp -= tp[0]

    write_to_csv('elevation_' + filename(height[n], period, angle[n], run[n]) + '.csv', t1, w1, w2)

    percent_change = (hmax2 - hmax1) / hmax1 * 100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot([0, 12], [0, 0], 'k:')
    plt.plot(t1, w1, lw=2, label='Waves in')
    plt.plot(t1, w2, lw=2, label='Waves out')
    plt.legend(loc='upper left')
    plt.xlabel(r'Time [$s$]')
    plt.ylabel(r'$\eta$ [$m$]')
    plt.xlim(0, 7)
    plt.ylim(-0.08, 0.16)
    plt.title(title(height[n], period, angle[n]))
    plt.grid()
    plt.savefig('elevation_time_series_%s.png' % filename(height[n], period, angle[n], run[n]))
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot([0, 12], [0, 0], 'k:')
    plt.plot(tp, pres1, lw=2, label='A')
    plt.plot(tp, pres2, lw=2, label='B')
    plt.plot(tp, pres3, lw=2, label='C')
    plt.plot(tp, pres4, lw=2, label='D')
    plt.legend(loc='upper left', ncol=4)
    plt.xlabel(r'Time [$s$]')
    plt.ylabel(r'$p$ [$hPa$]')
    plt.xlim(0, 7)
    plt.ylim(- 2.4, 2.4)
    plt.title(title(height[n], period, angle[n]))
    plt.grid()
    plt.savefig('pressure_time_series_%s.png' % filename(height[n], period, angle[n], run[n]))
    plt.close()

    csv = ','.join([
        angle[n],
        '%i' % height[n],
        '%i' % period,
        '%.3f' % hmax1,
        '%.3f' % hmax2,
        '%.3f' % percent_change
    ])

    f.write(csv + '\n')

    print(title(height[n], period, angle[n]))
    print(30 * '-')
    print('H_in: %.3f m, H_out: %.3f m' % (hmax1, hmax2))
    print('Percent change: %.2f' % percent_change + '%')
    print()

f.close()

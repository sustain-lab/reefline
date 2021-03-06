from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import numpy as np
from reefline.dispersion import w2k
from reefline.wavewire import read_wavewire_from_toa5
from reefline.udm import read_udm_from_toa5
from reefline.utility import power_spectrum, running_mean, write_to_csv
from scipy.signal import detrend
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

f = np.array([
    0.64, 0.45, 0.32, 0.32, 0.64,
    0.45, 0.45, 0.89, 0.89, 0.64 
])

depth = np.array([
    0.32, 0.32, 0.32, 0.32, 0.32, # 1/20 scaling
    0.32, 0.32, 0.32, 0.32, 0.32, # 1/20 scaling
])

omega = 2 * np.pi * f
k = np.array([w2k(omega[n], h=depth[n])[0][0] for n in range(f.size)])
Cp = omega / k

# all 30-second runs
runs_start = [

    datetime(2022, 2, 23, 15, 32,  6), # a = 0.03, f = 0.64, orthogonal
    datetime(2022, 2, 23, 15, 49, 20), # a = 0.06, f = 0.45, orthogonal
    datetime(2022, 2, 23, 17, 21, 10), # a = 0.06, f = 0.32, 10 deg. oblique
    datetime(2022, 2, 23, 19,  2, 15), # a = 0.06, f = 0.32, 10 deg. oblique
    datetime(2022, 2, 23, 19, 24, 20), # a = 0.03, f = 0.64, 30 deg. oblique
    datetime(2022, 2, 23, 19, 29, 15), # a = 0.06, f = 0.45, 30 deg. oblique
    datetime(2022, 2, 23, 19, 31, 30), # a = 0.06, f = 0.45, 30 deg. oblique
    datetime(2022, 2, 23, 19, 47, 52), # a = 0.03, f = 0.89, 45 deg. oblique
    datetime(2022, 2, 23, 19, 49, 30), # a = 0.03, f = 0.89, 45 deg. oblique
    datetime(2022, 2, 23, 19, 52, 10), # a = 0.06, f = 0.64, 45 deg. oblique

]

titles = [
    "Hs = 4 ft, Tp = 7 s, orthogonal",
    "Hs = 8 ft, Tp = 10 s, orthogonal",
    "Hs = 8 ft, Tp = 14 s, 10 deg. oblique",
    "Hs = 8 ft, Tp = 14 s, 10 deg. oblique",
    "Hs = 4 ft, Tp = 7 s, 30 deg. oblique",
    "Hs = 8 ft, Tp = 10 s, 30 deg. oblique",
    "Hs = 8 ft, Tp = 10 s, 30 deg. oblique",
    "Hs = 4 ft, Tp = 5 s, 45 deg. oblique",
    "Hs = 4 ft, Tp = 5 s, 45 deg. oblique",
    "Hs = 8 ft, Tp = 7 s, 45 deg. oblique"
]

filenames = [
    #'4ft_7s_baseline',
    #'8ft_10s_baseline',
    #'4ft_5s_baseline',
    #'8ft_7s_baseline',
    #'8ft_14s_baseline',
    '4ft_7s_orthogonal',
    '8ft_10s_orthogonal',
    '8ft_14s_10deg_a',
    '8ft_14s_10deg_b',
    '4ft_7s_30deg',
    '8ft_10s_30deg_a',
    '8ft_10s_30deg_b',
    '4ft_5s_45deg_a',
    '4ft_5s_45deg_b',
    '8ft_7s_45deg',
]

height = [#4, 8, 4, 8, 8,
    4, 8, 8, 8, 4, 8, 8, 4, 4, 8]
period = [#7, 10, 5, 7, 14,
    7, 10, 14, 14, 7, 10, 10, 5, 5, 7]

experiment = [
    #'baseline',
    #'baseline',
    #'baseline',
    #'baseline',
    #'baseline',
    'model_orthogonal',
    'model_orthogonal',
    'model_10deg',
    'model_10deg',
    'model_30deg',
    'model_30deg',
    'model_30deg',
    'model_45deg',
    'model_45deg',
    'model_45deg',
]

run_seconds = 30

udm_files = glob.glob('data/udm/TOA5_SUSTAIN_ELEVx6d*')
udm_files.sort()
time, u1, u2, u3, u4, u5, u6 = read_udm_from_toa5(udm_files)

num_sensors = 6
num_runs = len(runs_start)
num_records = run_seconds * 20 + 1

e1 = np.zeros((num_runs, num_records))
e2 = np.zeros((num_runs, num_records))

for n, t1 in enumerate(runs_start):
    t2 = t1 + timedelta(seconds=run_seconds)
    mask = (time >= t1) & (time <= t2)
    e1[n,:] = detrend(- u1[mask])
    e2[n,:] = detrend(- u5[mask])

udm_mask = [0, 2, 4]
x = np.array([-0.63, 0.31, 2.14])
x -= x[1]

dx = x[2] - x[0] # distance between UDM out and UDM in
time_shift = dx / Cp # time shift for UDM out

f = open('reefline.csv', 'w')
f.write('experiment,height,period,measured_height_in,measured_height_out,percent_change\n')

for n in range(num_runs):

    t = np.linspace(0, 30, num_records)
    mask1 = (t >= 5) & (t < 12)
    t1 = t[mask1]

    t2 = np.round(t - time_shift[n], 2)
    mask2 = (t2 >= 5) & (t2 < 12)
    t2 = t2[mask2]

    e1s = running_mean(e1[n,:], 3)
    e2s = running_mean(e2[n,:], 3)

    w1 = detrend(e1s[mask1])
    w2 = detrend(e2s[mask2])

    hmax1 = np.max(w1) - np.min(w1)
    hmax2 = np.max(w2) - np.min(w2)
    t1 -= t1[0]

    write_to_csv('elevation_' + filenames[n] + '.csv', t1, w1, w2)

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
    plt.ylim(-0.06, 0.12)
    plt.title(titles[n])
    plt.grid()
    plt.savefig('time_series_%s.png' % filenames[n])
    plt.close()

    csv = ','.join([
        experiment[n],
        '%i' % height[n],
        '%i' % period[n],
        '%.3f' % hmax1,
        '%.3f' % hmax2,
        '%.3f' % percent_change
    ])

    f.write(csv + '\n')

    print(titles[n])
    print(30 * '-')
    print('H_in: %.3f m, H_out: %.3f m' % (hmax1, hmax2))
    print('Percent change: %.2f' % percent_change + '%')
    print()

f.close()

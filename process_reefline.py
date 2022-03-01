from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import numpy as np
from reefline.wavewire import read_wavewire_from_toa5
from reefline.udm import read_udm_from_toa5
from reefline.utility import power_spectrum
from scipy.signal import detrend

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

run_seconds = 30

udm_files = glob.glob('data/TOA5_SUSTAIN_ELEVx6d*')
udm_files.sort()
time, u1, u2, u3, u4, u5, u6 = read_udm_from_toa5(udm_files)

ww_files = glob.glob('data/TOA5_OSSwaveX4.elev*')
ww_files.sort()
#_, data = read_wavewire_from_toa5(ww_files)
#w1 = data['w4'] # wire 4 is on the inlet side
#w2 = data['w3'] # wire 3 is on the beach side


num_sensors = 6
num_runs = len(runs_start)
num_records = run_seconds * 20 + 1

eta = np.zeros((num_runs, num_sensors, num_records))

for n, t1 in enumerate(runs_start):
    t2 = t1 + timedelta(seconds=run_seconds)
    mask = (time >= t1) & (time <= t2)
    eta[n,0,:] = detrend(- u1[mask])
    eta[n,1,:] = detrend(- u2[mask])
    eta[n,2,:] = detrend(- u3[mask])
    eta[n,3,:] = detrend(- u4[mask])
    eta[n,4,:] = detrend(- u5[mask])
    eta[n,5,:] = detrend(- u6[mask])

t = np.linspace(0, 30, num_records)

rms_mask = (t > 5) & (t < 15)

eta[eta < -0.1] = 0
Hrms = np.sqrt(np.mean((2 * eta[:,:,rms_mask])**2, axis=2))
udm_mask = [0, 2, 4]
x = np.array([-0.63, 0.31, 2.14])
x -= x[1]

for n in range(num_runs):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    F1, f = power_spectrum(eta[n,0], 1 / 20)
    F2, f = power_spectrum(eta[n,4], 1 / 20)
    plt.loglog(f, F1, lw=1, label='Waves in')
    plt.loglog(f, F2, lw=1, label='Waves out')
    plt.legend(loc='upper left')
    plt.xlabel(r'$f$ [$Hz$]')
    plt.ylabel(r'$S_{\eta\eta}$ [$m^2/Hz$]')
    plt.title(titles[n])
    plt.grid()
    plt.xlim(0.1, 10)
    plt.savefig('spectrum_%2.2i.png' % n)
    plt.close()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot(t, eta[n,0], lw=1, label='Waves in')
    plt.plot(t, eta[n,2], lw=1, label='Waves center')
    plt.plot(t, eta[n,4], lw=1, label='Waves out')
    plt.legend(loc='upper left')
    plt.xlabel(r'Time [$s$]')
    plt.ylabel(r'$\eta$ [$m$]')
    plt.xlim(0, 30)
    plt.title(titles[n])
    plt.grid()
    plt.savefig('time_series_%2.2i.png' % n)
    plt.close()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    plt.plot(x, Hrms[n,udm_mask], marker='o')
    plt.xlabel(r'Shoreward distance from reef [$m$]')
    plt.ylabel(r'$H_{RMS}$ [$m$]')
    plt.title(titles[n])
    plt.grid()
    plt.savefig('Hrms_%2.2i.png' % n)
    plt.close()

from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from reefline.aquadopp import read_aquadopp_time, read_aquadopp_velocity, cell_distance
from reefline.dispersion import w2k
from reefline.pressure import read_pressure_from_toa5
from reefline.udm import read_udm_from_toa5
from reefline.utility import power_spectrum, running_mean, write_to_csv
from scipy.signal import detrend
import matplotlib

matplotlib.rcParams.update({'font.size': 14})
    
von_karman = 0.4

def compute_drag_from_profile(time, u, v, z, start_time, end_time):
    """Computes friction velocity and drag coefficient
    from mean current profile."""
    mask = (time > runs_start[n]) & (time <= runs_start[n] + run_duration)
    vel_mean = np.mean(np.sqrt(u**2 + v**2)[mask].T, axis=1)
    fit = np.polyfit(z, vel_mean, 1)
    logfit = np.polyfit(np.log(z), vel_mean, 1)
    ust = von_karman * logfit[0]
    U = np.mean(vel_mean)
    CD = ust**2 / U**2
    return U, ust, CD


f = 0.38
depth = 0.44
omega = 2 * np.pi * f
#k = np.array([w2k(omega[n], h=depth[n])[0][0] for n in range(f.size)])
k = w2k(omega, h=depth)[0][0]
Cp = omega / k

height = [4, 4, 4, 8, 8, 8, 4, 4, 4, 8, 8, 8]
run = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
period = 14
angle = 6 * ['orthogonal'] + 6 * ['45 deg. oblique']
amplitude = np.array(2 * (3 * [0.03] + 3 * [0.06]))

# all 30-second runs
runs_start = [

    datetime(2022, 3, 9, 17,  5, 12), # a = 0.03, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17,  9, 12), # SKIP a = 0.03, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 13, 24), # SKIP a = 0.03, f = 0.38, orthogonal

    datetime(2022, 3, 9, 17, 17, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 21, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 25, 12), # a = 0.06, f = 0.38, orthogonal

    datetime(2022, 3, 9, 17, 37, 12), # a = 0.03, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 41, 12), # a = 0.03, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 45, 12), # a = 0.03, f = 0.38, 45 deg. oblique

    datetime(2022, 3, 9, 17, 49, 12), # SKIP a = 0.06, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 53, 12), # a = 0.06, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 57, 12), # a = 0.06, f = 0.38, 45 deg. oblique
]

# skip runs that didn't have full aquadopp coverage,
# due to gaps between bursts
runs_to_skip = [1, 2, 9]

run_seconds = 30

aquadopp1_path = '/home/milan/Work/sustain/reefline/data/Aquadopp/BASL705' # upwave
aquadopp2_path = '/home/milan/Work/sustain/reefline/data/Aquadopp/BASL605' # downwave

time1 = read_aquadopp_time(aquadopp1_path + '.sen')
time2 = read_aquadopp_time(aquadopp2_path + '.sen')

u1 = np.clip(read_aquadopp_velocity(aquadopp1_path + '.v1'), -0.499, 0.499)
v1 = np.clip(read_aquadopp_velocity(aquadopp1_path + '.v2'), -0.499, 0.499)
w1 = np.clip(read_aquadopp_velocity(aquadopp1_path + '.v3'), -0.499, 0.499)

u2 = np.clip(read_aquadopp_velocity(aquadopp2_path + '.v1'), -0.499, 0.499)
v2 = np.clip(read_aquadopp_velocity(aquadopp2_path + '.v2'), -0.499, 0.499)
w2 = np.clip(read_aquadopp_velocity(aquadopp2_path + '.v3'), -0.499, 0.499)

run_duration = timedelta(seconds=20)

z = np.array(cell_distance) + 0.1 # 0.1 m is the offset of sensor head from the bottom

U1, ust1, CD1 = [], [], [] # upwave
U2, ust2, CD2 = [], [], [] # downwave
Hs = []

for n in range(len(runs_start)):

    start_time = runs_start[n]
    end_time = start_time + run_duration

    if n in runs_to_skip:
        continue

    U, ust, CD = compute_drag_from_profile(time1, u1, v1, z, start_time, end_time)
    U1.append(U)
    ust1.append(ust)
    CD1.append(CD)

    U, ust, CD = compute_drag_from_profile(time2, u2, v2, z, start_time, end_time)
    U2.append(U)
    ust2.append(ust)
    CD2.append(CD)

    Hs.append(height[n])

Hs = np.array(Hs)
U1 = np.array(U1)
ust1 = np.array(ust1)
CD1 = np.array(CD1)
U2 = np.array(U2)
ust2 = np.array(ust2)
CD2 = np.array(CD2)

fig = plt.figure(figsize=(8, 6))
plt.plot(U1[Hs == 4], ust1[Hs == 4], marker='o', linestyle='', ms=8, mec='k', color='tab:blue', label='Upwave, 4 ft')
plt.plot(U1[Hs == 8], ust1[Hs == 8], marker='*', linestyle='', ms=12, mec='k', color='tab:blue', label='Upwave, 8 ft')
plt.plot(U2[Hs == 4], ust2[Hs == 4], marker='o', linestyle='', ms=8, mec='k', color='tab:orange', label='Downwave, 4 ft')
plt.plot(U2[Hs == 8], ust2[Hs == 8], marker='*', linestyle='', ms=12, mec='k', color='tab:orange', label='Downwave, 8 ft')
plt.legend(loc='upper left', shadow=True, fancybox=True)
plt.grid()
plt.xlabel('U [m/s]')
plt.ylabel(r'$u_*$ [m/s]')
plt.xlim(0, 0.3)
plt.ylim(0, 0.12)
plt.title('Friction velocity at 14-26 cm from the bottom')
plt.savefig('ust.png', dpi=150)
plt.close()

fig = plt.figure(figsize=(8, 6))
plt.plot(U1[Hs == 4], CD1[Hs == 4], marker='o', linestyle='', ms=8, mec='k', color='tab:blue', label='Upwave, 4 ft')
plt.plot(U1[Hs == 8], CD1[Hs == 8], marker='*', linestyle='', ms=12, mec='k', color='tab:blue', label='Upwave, 8 ft')
plt.plot(U2[Hs == 4], CD2[Hs == 4], marker='o', linestyle='', ms=8, mec='k', color='tab:orange', label='Downwave, 4 ft')
plt.plot(U2[Hs == 8], CD2[Hs == 8], marker='*', linestyle='', ms=12, mec='k', color='tab:orange', label='Downwave, 8 ft')
plt.legend(loc='upper left', shadow=True, fancybox=True)
plt.grid()
plt.xlabel('U [m/s]')
plt.ylabel(r'$C_D$ [m/s]')
plt.xlim(0, 0.3)
plt.ylim(0, 0.5)
plt.title('Drag coefficient at 14-26 cm from the bottom')  
plt.savefig('cd.png', dpi=150)
plt.close()

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

def plot_velocity_profile(time, u, v, w, z, elev_time, elev, start_time, end_time, \
    height, period, angle, run, position):

    fig = plt.figure(figsize=(8, 12))
    ax1, ax2, ax3, ax4 = axes = [
        plt.subplot2grid((4, 1), (0, 0)),
        plt.subplot2grid((4, 1), (1, 0)),
        plt.subplot2grid((4, 1), (2, 0)),
        plt.subplot2grid((4, 1), (3, 0))
    ]

    im = ax1.contourf(time, z, v.T, np.arange(-0.5, 0.55, 0.05), cmap=cm.bwr)
    im = ax2.contourf(time, z, u.T, np.arange(-0.5, 0.55, 0.05), cmap=cm.bwr)
    im = ax3.contourf(time, z, w.T, np.arange(-0.5, 0.55, 0.05), cmap=cm.bwr)
    fig.colorbar(im, ax=ax3, shrink=0.8, ticks=np.arange(-0.5, 0.6, 0.1), location='bottom')
    
    ax1.contour(time, z, v.T, [0], linewidths=0.2, colors='k')
    ax2.contour(time, z, u.T, [0], linewidths=0.2, colors='k')
    ax3.contour(time, z, w.T, [0], linewidths=0.2, colors='k')

    ax4.plot(elev_time, elev, 'k-', label='Elevation')
    ax4.plot(time, np.mean(v, axis=1), label='u-vel.')
    ax4.plot(time, np.mean(u, axis=1), label='v-vel.')
    ax4.plot(time, np.mean(w, axis=1), label='w-vel.')
    ax4.legend(ncol=4)
    ax4.grid()

    # Plot theoretical surface vel. based on linear wave theory
    u_max = omega * amplitude[n]
    ax4.plot([runs_start[n], runs_start[n] + run_duration], [u_max, u_max], 'k--')
    ax4.plot([runs_start[n], runs_start[n] + run_duration], [-u_max, -u_max], 'k--')

    for ax in axes:
        ax.set_xlim(runs_start[n], runs_start[n] + run_duration)
    
    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_ylabel('z [m]')
    
    ax4.set_ylabel(r'$\eta$ [m], v [m/s]')
    ax4.set_ylim(-0.3, 0.3)

    ax1.set_title('Along-tank velocity [m/s]')
    ax2.set_title('Cross-tank velocity [m/s]')
    ax3.set_title('Vertical velocity [m/s]')

    ax4.set_xlabel('Time [UTC]')
    fig.suptitle(position + ' ' + title(height, period, angle))
    plt.savefig('velocity_' + position.lower() + '_%s.png' % filename(height, period, angle, run))
    plt.close()


def plot_mean_velocity_profile(time, u, v, w, z, elev_time, elev, start_time, end_time, \
    height, period, angle, run, position):

    mask = (time > runs_start[n]) & (time <= runs_start[n] + run_duration)
    vel_mean = np.mean(np.sqrt(u**2 + v**2)[mask].T, axis=1)
    fit = np.polyfit(z, vel_mean, 1)
    logfit = np.polyfit(np.log(z), vel_mean, 1)
    ust = von_karman * logfit[0]
    U = np.mean(vel_mean)
    CD = ust**2 / U**2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.semilogy(vel_mean, z, marker='o')
    ax.semilogy(np.polyval(fit, z), z, 'k--')
    ax.grid()
    ax.set_xlabel('Horizontal velocity [m/s]')
    ax.set_ylabel('Distance from depth [m]')
    ax.text(0.02, 0.94, r'$U$ = %.2f m/s' % U, ha='left', va='bottom',
            transform=ax.transAxes, fontsize=16)
    ax.text(0.02, 0.88, r'$u_*$ = %.2f m/s' % ust, ha='left', va='bottom',
            transform=ax.transAxes, fontsize=16)
    ax.text(0.02, 0.82, r'$C_D$ = %.3f' % CD, ha='left', va='bottom',
            transform=ax.transAxes, fontsize=16)
    ax.set_xlim(0, 0.4)
    fig.subplots_adjust(left=0.2)
    fig.suptitle(position + ' ' + title(height, period, angle))
    plt.savefig('mean_profile_' + position.lower() + '_%s.png' % filename(height, period, angle, run))
    plt.close()


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
    datetime(2022, 3, 9, 17,  9, 12), # a = 0.03, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 13, 24), # a = 0.03, f = 0.38, orthogonal

    datetime(2022, 3, 9, 17, 17, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 21, 12), # a = 0.06, f = 0.38, orthogonal
    datetime(2022, 3, 9, 17, 25, 12), # a = 0.06, f = 0.38, orthogonal

    datetime(2022, 3, 9, 17, 37, 12), # a = 0.03, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 41, 12), # a = 0.03, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 45, 12), # a = 0.03, f = 0.38, 45 deg. oblique

    datetime(2022, 3, 9, 17, 49, 12), # a = 0.06, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 53, 12), # a = 0.06, f = 0.38, 45 deg. oblique
    datetime(2022, 3, 9, 17, 57, 12), # a = 0.06, f = 0.38, 45 deg. oblique
]

run_seconds = 30

# Elevation from UDM
udm_files = glob.glob('data/udm/TOA5_SUSTAIN_ELEVx6d*2022_03_09*')
udm_files.sort()
time, u1, u2, u3, u4, u5, u6 = read_udm_from_toa5(udm_files)

for n in range(3):
    udm1 = - detrend(despike(u1))
    udm6 = - detrend(despike(u6))

num_runs = len(runs_start)
num_records = run_seconds * 20 + 1
num_records_pressure = run_seconds * 10 + 1

e1 = np.zeros((num_runs, num_records))
e2 = np.zeros((num_runs, num_records))

for n, t1 in enumerate(runs_start):
    t2 = t1 + timedelta(seconds=run_seconds)

    mask = (time >= t1) & (time <= t2)
    e1[n,:] = detrend(udm1[mask])
    e2[n,:] = detrend(udm6[mask])

elev1 = udm1[:]
elev2 = udm6[:]

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

vel1 = np.sqrt(u1**2 + v1**2)
vel2 = np.sqrt(u2**2 + v2**2)

run_duration = timedelta(seconds=20)

z = np.array(cell_distance) + 0.1 # 0.1 m is the offset of sensor head from the bottom

for n in range(num_runs):
    plot_velocity_profile(time1, u1, v1, w1, z, time, elev1, \
                          runs_start[n], runs_start[n] + run_duration, \
                          height[n], period, angle[n], run[n], 'upwave')
    plot_velocity_profile(time2, u2, v2, w2, z, time, elev2, \
                          runs_start[n], runs_start[n] + run_duration, \
                          height[n], period, angle[n], run[n], 'downwave')

    plot_mean_velocity_profile(time1, u1, v1, w1, z, time, elev1, \
                          runs_start[n], runs_start[n] + run_duration, \
                          height[n], period, angle[n], run[n], 'upwave')
    plot_mean_velocity_profile(time2, u2, v2, w2, z, time, elev2, \
                          runs_start[n], runs_start[n] + run_duration, \
                          height[n], period, angle[n], run[n], 'downwave')

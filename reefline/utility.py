"""
clasi.utility -- Various utility functions used to process data.
"""
import numpy as np
from matplotlib.mlab import detrend_linear


def binavg(x, binsize):
    """Simple binned average over binsize elements."""
    return np.array([np.mean(x[n:n+binsize])\
        for n in range(0, len(x), binsize)])


def blackman_harris(n):
    """Returns the n-point Blackman-Harris window."""
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = [0.35875, -0.48829, 0.14128, -0.01168]
    res = np.zeros((n))
    for i in range(4):
        res += p[i] * np.cos(i * x)
    return res


def cat_numpy_arrays(arrays):
    """Concatenates numpy arrays."""
    x = []
    for a in arrays:
        x += list(a)
    return np.array(x)


def limit_to_percentile_range(x, plow, phigh):
    """Limits the values of x to low and high percentile limits."""
    xlow, xhigh = np.percentile(x, plow), np.percentile(x, phigh)
    x[x < xlow] = xlow
    x[x > xhigh] = xhigh
    return x


def filter_by_frequency(x, dt, fmin=None, fmax=None):
    """
    Filters x in frequency space.
    dt is the time step.
    fmin and fmax are minimum and maximum cutoff frequencies.
    """
    f = np.fft.fftfreq(x.size, dt)
    s = np.fft.fft(x)
    if fmin:
        s[(f < fmin) & (f > - fmin)] = 0
    if fmax:
        s[(f > fmax) | (f < - fmax)] = 0
    return np.fft.ifft(s).real


def integrate_by_frequency(x, dt, fmin=None, fmax=None):
    """
    Integrates x in frequency space.
    dt is the time step.
    fmin and fmax are minimum and maximum cutoff frequencies.
    """
    f = np.fft.fftfreq(x.size, dt)
    s = np.fft.fft(x)
    s[1:] /= (1j * 2 * np.pi * f[1:])
    if fmin:
        s[(f < fmin) & (f > - fmin)] = 0
    if fmax:
        s[(f > fmax) | (f < - fmax)] = 0
    return np.fft.ifft(s).real


def running_mean(x, n):
    """Running mean with the window n."""
    return np.convolve(x, np.ones((n,)) / n, mode='same')


def power_spectrum(x, dt, binsize=1):
    """Power spectrum of x with a sampling interval dt.
    Optionally, average over binsize if provided."""

    assert dt > 0, 'dt must be > 0'
    assert type(binsize) is int, 'binsize must be an int'
    assert binsize > 0, 'binsize must be > 0'

    N = x.size
    window = blackman_harris(N)
    Sx = np.fft.fft(window * detrend_linear(x))[:N//2]
    C = dt / (np.pi * np.sum(window**2))
    df = 2 * np.pi / (dt * N)
    f = np.array([i * df for i in range(N//2)]) / (2 * np.pi)

    if binsize > 1:
        Sxx = 2 * np.pi * C * binavg(np.abs(Sx)**2, binsize)
        f = binavg(f, binsize)
        df *= binsize
    else:
        Sxx = 2 * np.pi * C * np.abs(Sx)**2

    return Sxx.astype(np.float32), f.astype(np.float32)


def cross_spectrum(x, y, dt, binsize=1):
    """Cross spectrum of x and y with a sampling interval dt.
    Optionally, average over binsize if provided."""

    assert x.size == y.size, 'x and y must have same size'
    assert dt > 0, 'dt must be > 0'
    assert type(binsize) is int, 'binsize must be an int'
    assert binsize > 0, 'binsize must be > 0'

    N = x.size
    window = blackman_harris(N)
    Sx = np.fft.fft(window * detrend_linear(x))[:N//2]
    Sy = np.fft.fft(window * detrend_linear(y))[:N//2]
    df = 2 * np.pi / (dt * N)
    f = np.array([i * df for i in range(N//2)]) / (2 * np.pi)
    C = dt / (np.pi * np.sum(window**2))

    if binsize > 1:
        Sxx = 2 * np.pi * C * binavg(np.abs(Sx)**2, binsize)
        Syy = 2 * np.pi * C * binavg(np.abs(Sy)**2, binsize)
        Sxy = 2 * np.pi * C * binavg(np.conj(Sx) * Sy, binsize)
        f = binavg(f, binsize)
        df *= binsize
    else:
        Sxx = 2 * np.pi * C * np.abs(Sx)**2
        Syy = 2 * np.pi * C * np.abs(Sy)**2
        Sxy = 2 * np.pi * C * np.conj(Sx) * Sy

    phase = np.arctan2(-np.imag(Sxy), np.real(Sxy))
    coherence = np.abs(Sxy / np.sqrt(Sxx * Syy))

    return Sxx.astype(np.float32), Syy.astype(np.float32), Sxy.astype(np.float32), \
        phase.astype(np.float32), coherence.astype(np.float32), f.astype(np.float32)


def read_coare35(filename):
    """Loads data that the COARE3.5 algorithm is based on, excluding ship data.
    Described by Edson et al. (2013), provided by Chris Fairall."""
    f = [x.strip().split() for x in open(filename).readlines()][1:]
    wspd = np.array([float(x[0]) for x in f])
    cd = np.array([float(x[1]) for x in f])
    cd_std = np.array([float(x[2]) for x in f])
    cd_count = np.array([int(float(x[3]) * 1e3) for x in f])
    return wspd, cd, cd_std, cd_count


def rotate(u, w, th):
    """Rotates the vector (u, w) by angle th."""
    ur =   np.cos(th) * u + np.sin(th) * w
    wr = - np.sin(th) * u + np.cos(th) * w
    return ur, wr


def scale_to_height(ust, Uz, z, zref):
    """Scales wind Uz at height z to zref."""
    VON_KARMAN = 0.4
    return Uz + ust / VON_KARMAN * np.log(zref / z)

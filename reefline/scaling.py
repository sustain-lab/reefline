from reefline.dispersion import w2k, k2w
import numpy as np

def scale_to_sustain(Hs, Tp, depth, scale_factor):
    omega = 2 * np.pi / Tp
    k = w2k(omega, h=depth)[0][0]
    wavelength = 2 * np.pi / k
    depth_scaled = depth * scale_factor
    wavelength_scaled = wavelength * scale_factor
    k_scaled = 2 * np.pi / wavelength_scaled
    omega_scaled = k2w(k_scaled, h=depth_scaled)[0][0]
    Tp_scaled = 2 * np.pi / omega_scaled
    Hs_scaled = Hs * scale_factor
    return Hs_scaled, Tp_scaled

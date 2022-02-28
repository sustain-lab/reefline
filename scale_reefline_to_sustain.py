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


scale_factor = 1 / 20
f2m = 1 / 3.28084

depth = 21 # feet, no surge
#depth = 29 # feet, surge

conditions = [
    {'Hs': 4, 'Tp': 7},  # Cold fronts (30 degrees)
    {'Hs': 8, 'Tp': 10}, # Cold fronts (30 degrees)
    {'Hs': 4, 'Tp': 5},  # Onshore 45 degrees
    {'Hs': 8, 'Tp': 7},  # Onshore 45 degrees
    {'Hs': 8, 'Tp': 14}  # Nor'easters (10 degrees)
]

for case in conditions:
    print('Hs = %i ft, Tp = %i s' % (case['Hs'], case['Tp']))
    print(20 * '-')
    Hs_scaled, Tp_scaled = scale_to_sustain(case['Hs'] * f2m, case['Tp'], depth * f2m, scale_factor)
    print('Hs_scaled = %.2f, f_scaled = %.2f' % (Hs_scaled, 1 / Tp_scaled))
    print()

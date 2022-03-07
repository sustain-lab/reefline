from reefline.scaling import scale_to_sustain
import numpy as np

scale_factor = 1 / 20
f2m = 1 / 3.28084

#depth = 21 # feet, no surge
depth = 29 # feet, surge

conditions = [
    {'Hs': 8, 'Tp': 11.7},
    {'Hs': 12, 'Tp': 11.7},
    {'Hs': 16, 'Tp': 11.7},
    {'Hs': 20, 'Tp': 11.7},
]

for case in conditions:
    print('Hs = %i ft, Tp = %i s' % (case['Hs'], case['Tp']))
    print(20 * '-')
    Hs_scaled, Tp_scaled = scale_to_sustain(case['Hs'] * f2m, case['Tp'], depth * f2m, scale_factor)
    print('Hs_scaled = %.2f, f_scaled = %.2f' % (Hs_scaled, 1 / Tp_scaled))
    print()

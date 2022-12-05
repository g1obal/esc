"""
Compare data obtained from
new and old program versions

Author: Gokhan Oztarhan
Created date: 04/12/2022
Last modified: 04/12/2022
"""

import os

import numpy as np


OUTPUT_FILE_NAME = 'output'


def get_parent_dir(root):
    return os.path.normpath(root).split(os.sep)[-2]


def get_data(root_dir):
    energy = {}
    coef = {}
    for root, dirs, files in sorted(os.walk(root_dir)):
        if OUTPUT_FILE_NAME in files:
            parent = get_parent_dir(root)
            
            with open(os.path.join(root, OUTPUT_FILE_NAME), 'r') as f:
                lines = f.readlines()
            
                for line in lines[-1::-1]:
                    if 'E_total' in line:
                        energy[parent] = line.split()[-2]
                        break

            coef[parent] = np.loadtxt(os.path.join(root, 'orb_dot_coef'))
        
    return energy, coef
    

energy_new, coef_new = get_data('new')
energy_old, coef_old = get_data('old')

print('Energy differences if diff != 0')
for key in energy_new:
    diff = float(energy_new[key]) - float(energy_old[key])
    if diff != 0:
        print(key, diff)
        
print('\n\nOrbital coef. differences if diff != 0')
for key in coef_new:
    diff = np.sqrt(((coef_new[key] - coef_old[key])**2).sum())
    if diff != 0:
        print(key, diff)

        

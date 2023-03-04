"""
Compare data obtained from
new and old program versions

Author: Gokhan Oztarhan
Created date: 04/12/2022
Last modified: 03/03/2023
"""

import os

import numpy as np
import pandas as pd


OUTPUT_FILE_NAME = 'out'

ROOT_DIR_NEW = 'new'
ROOT_DIR_OLD = 'old'

PATH_NEW = {}
PATH_OLD = {}

for root, dirs, files in sorted(os.walk(ROOT_DIR_NEW)):
    if OUTPUT_FILE_NAME in files:
        parent = os.path.normpath(root).split(os.sep)[-1]
        PATH_NEW[parent] = root

for root, dirs, files in sorted(os.walk(ROOT_DIR_OLD)):
    if OUTPUT_FILE_NAME in files:
        parent = os.path.normpath(root).split(os.sep)[-1]
        PATH_OLD[parent] = root


def get_data(root):   
    with open(os.path.join(root, OUTPUT_FILE_NAME), 'r') as f:
        lines = f.readlines()
        
        E_total = np.nan
        for line in lines[-1::-1]:
            if 'E_total_nau' in line:
                E_total = float(line.split()[-2])
                break
    
    data = np.load(os.path.join(root, 'data.npz'))
    if data['mode'] == 'tb':
        E_up, E_dn = data['E'], data['E']
        V_up, V_dn = data['V'], data['V']
    elif data['mode'] == 'mfh':
        E_up, E_dn = data['E_up'], data['E_dn']
        V_up, V_dn = data['V_up'], data['V_dn']
    
    coef = np.loadtxt(os.path.join(root, 'orb_dot_coef'))
    
    return E_total, E_up, E_dn, V_up, V_dn, coef


# Initialize main DataFrame
df = pd.DataFrame()

for key in PATH_NEW:
    E_total_new, E_up_new, E_dn_new, V_up_new, V_dn_new, coef_new = \
        get_data(PATH_NEW[key])
    E_total_old, E_up_old, E_dn_old, V_up_old, V_dn_old, coef_old = \
        get_data(PATH_OLD[key])                 
    
    # Energy difference
    E_diff = E_total_new - E_total_old
    
    # Orbital coefficient difference
    coef_diff = np.sqrt(((coef_new - coef_old)**2).sum())

    # Eigenvector comparison
    # ** If an eigenvector in the new array is parallel to an eigenvector in 
    # the old array (which means inner product of them will be 1), it is 
    # orthogonal to the other eigenvectors in the old array since old 
    # eigenvectors are already orthogonal to each other.
    #
    # ** If the number of orthogonal eigenvectors (between old and new matrices)
    # is not equal to the number of columns in the matrices, there may be 
    # degenerate states (which means there may be multiple eigenvalues equal to 
    # each other), or the matrices may not be orthogonal. If there are 
    # degenerate states, some eigenvectors may not be orthogonal.
    E_values, counts = np.unique(E_up_new.round(8), return_counts=True)
    degeneracy_up = (counts > 1).sum()

    n_vectors_up = V_up_new.shape[1]
    prod = V_up_new.T @ V_up_old
    orthogonal_finder = (np.abs(prod) > 0.99).sum(axis=0)
    n_orthogonal_vectors_up = (orthogonal_finder == 1).sum()
    orth_diff_up = n_vectors_up - n_orthogonal_vectors_up
    
    E_values, counts = np.unique(E_dn_new.round(8), return_counts=True)
    degeneracy_dn = (counts > 1).sum()
    
    n_vectors_dn = V_dn_new.shape[1]
    prod = V_dn_new.T @ V_dn_old
    orthogonal_finder = (np.abs(prod) > 0.99).sum(axis=0)
    n_orthogonal_vectors_dn = (orthogonal_finder == 1).sum()
    orth_diff_dn = n_vectors_dn - n_orthogonal_vectors_dn

    #np.savetxt('prod', prod, fmt='% .5f', delimiter=' ', newline='\n')
    #input()
    
    # Add data to dict
    data = {
        'key': int(key),
        'PATH_NEW': PATH_NEW[key],
        'PATH_OLD': PATH_OLD[key],
        'coef_diff': coef_diff,
        'E_diff': E_diff,
        'orth_diff_up': orth_diff_up,
        'orth_diff_dn': orth_diff_dn,
        'n_vectors_up': n_vectors_up,
        'n_vectors_dn': n_vectors_dn,
        'n_orth_vec_up': n_orthogonal_vectors_up,
        'n_orth_vec_dn': n_orthogonal_vectors_dn,
        'degeneracy_up': degeneracy_up,
        'degeneracy_dn': degeneracy_dn,
    }
    
    # Convert dict to DataFrame
    df_temp = pd.DataFrame(data, index=[0])
    
    # Append values to main DataFrame
    df = pd.concat([df, df_temp], ignore_index=True, axis=0)
    
    print('key: %s\nPATH_NEW = %s\nPATH_OLD = %s\nDone.\n' \
        %(key, PATH_NEW[key], PATH_OLD[key]))

# Save to csv file
df = df.sort_values('key')
df.to_csv('data_e.csv', header=True, index=False, float_format='% .6e')
#df.to_csv('data_f.csv', header=True, index=False, float_format='% .6f')



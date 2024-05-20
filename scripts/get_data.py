"""
Data Reader

Author: Gokhan Oztarhan
Created date: 02/04/2024
Last modified: 02/04/2024
"""

import os
import time

import numpy as np
import pandas as pd


ROOT_DIR = '.'
OUTPUT_FILE_NAME = 'output'


def get_data():
    tic = time.time()
    
    # Form an empty dataframe
    columns = [
        'mode', 'U/t', 
        'n_side', 'n_elec', 'n_up', 'n_dn', 'E_total', 'p_edge_pol',
    ]
    df = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(ROOT_DIR):
        dirs.sort()
        if OUTPUT_FILE_NAME in files:    
            # Parse output file
            with open(os.path.join(root, OUTPUT_FILE_NAME), 'r') as f:
                lines = f.readlines()

            data = {
                'mode': get_feature('mode =', -1, str, lines),
                'U/t': get_feature('U/t =', -1, float, lines),
                'n_side': get_feature('n_side =', -1, int, lines),
                'n_elec': get_feature('n_elec =', -1, int, lines),
                'n_up': get_feature('n_up =', -1, int, lines),
                'n_dn': get_feature('n_dn =', -1, int, lines),
                'E_total': get_feature('E_total =', -1, float, lines),
                'p_edge_pol': get_feature('p_edge_pol =', -1, float, lines),
            }
            
            if np.isnan(data['U/t']):
                data['U/t'] = 0
            
            # Append to dataframe
            df = pd.concat(
                [df, pd.DataFrame(data, index=[0])], ignore_index=True, axis=0
            )
            
            # Print info
            print('Done: %s' %os.path.split(root)[-1])
    
    # Save dataframes to csv
    df.to_csv('data.csv', header=True, float_format='% .6f')

    toc = time.time() 
    print("Execution time, get_data = %.3f s" %(toc-tic))


def get_feature(string, ind, _type, lines):
    try:
        feature = _type([s for s in lines if string in s][0].split()[ind])
    except:
        feature = np.nan
    return feature


if __name__ == '__main__':
    get_data()

 

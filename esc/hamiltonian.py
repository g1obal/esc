"""
Hamiltonian Generator

Author: Gokhan Oztarhan
Created date: 20/07/2019
Last modified: 02/03/2023
"""

import time
import logging

import numpy as np


logger = logging.getLogger(__name__)


def hamiltonian_tb(t, tp, ind_NN, ind_NN_2nd, n_site):
    """
    Generates tight-binding hamiltonian matrix.
    
    Hamiltonian form:
        Htb = -t \sum_{<i,j>,\sigma} c_{i \sigma}^{\dagger} c_{j \sigma}
        where t is the 1st nearest neighbor hopping parameter.
        There is also a 2nd nearest neighbor hopping parameter, tp.
    
    See README.md for more information.
    """
    tic = time.time()
    
    H = np.zeros([n_site,n_site])
    
    # 1st nearest neighbor hopping
    H[ind_NN[:,0],ind_NN[:,1]] = -t
    H[ind_NN[:,1],ind_NN[:,0]] = -t
    
    # 2nd nearest neighbor hopping
    if tp != 0:
        H[ind_NN_2nd[:,0],ind_NN_2nd[:,1]] = -tp
        H[ind_NN_2nd[:,1],ind_NN_2nd[:,0]] = -tp
    
    memory_usage_H = H.nbytes / 1024 / 1024
    
    toc = time.time()

    string = 'Htb generated, %.3f MB. (%.3f s)\n\n' %(memory_usage_H, toc-tic)
    logger.info(string)
    
    return H


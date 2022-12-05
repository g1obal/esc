"""
Method
Tight-binding

Author: Gokhan Oztarhan
Created date: 29/08/2019
Last modified: 09/02/2021
"""

import time
import logging

import numpy as np
from numpy.linalg import eigh


logger = logging.getLogger(__name__)


def tb(Htb, n_up, n_dn):
    """Tight-binding"""
    # Eigenvalues and eigenvectors
    tic = time.time()
    E, V = eigh(Htb)
    toc = time.time()
    logger.info('eigh done. (%.3f s)\n' %(toc-tic))
            
    # Calculate up and dn rho (using same V)
    # Even though the Hamiltonians of up and down electrons are the same,
    # we need to calculate density matrix of up and down electrons separately. 
    rho_up = V[:,:n_up] @ V[:,:n_up].conj().T
    rho_dn = V[:,:n_dn] @ V[:,:n_dn].conj().T
    
    # Total energy 
    E_total = np.trace((rho_up + rho_dn) @ Htb)
    logger.info('E_total = %.15f eV\n\n' %(E_total))
            
    return E, V


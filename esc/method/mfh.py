"""
Method
Mean-field Hubbard

Author: Gokhan Oztarhan
Created date: 29/08/2019
Last modified: 10/01/2023
"""

import logging
import time
from copy import deepcopy

import numpy as np
from numpy.linalg import eigh


logger = logging.getLogger(__name__)


def mfh(
    Htb, U, mix_ratio, delta_E_lim, iter_lim, 
    n_up, n_dn, n_ave_up, n_ave_dn, U_LR=None
):
    """
    Self-consistent mean-field Hubbard
    
    Htb is the tight-binding Hamiltonian matrix (constructed in hamiltonian.py).
    See README.md for more information.
    """
    tic_mfh = time.time()
    
    # Mix_ratio
    mix_ratio_old = 1.0 - mix_ratio
    
    # Initialize E_total to infinity to skip first step
    E_total = np.inf

    # Self-consistent loop
    for i in range(iter_lim):
        # Construct the MFH Hamiltonian
        Hmfh_up = deepcopy(Htb)
        Hmfh_dn = deepcopy(Htb)
        Hmfh_up.ravel()[::Hmfh_up.shape[1]+1] += U * (n_ave_dn - 0.5)
        Hmfh_dn.ravel()[::Hmfh_dn.shape[1]+1] += U * (n_ave_up - 0.5) 
        if U_LR is not None:
            n_ave = (n_ave_up + n_ave_dn - 1)[:,None]
            U_LR_sum = 0.5 * U_LR @ n_ave
            Hmfh_up.ravel()[::Hmfh_up.shape[1]+1] += U_LR_sum[:,0]
            Hmfh_dn.ravel()[::Hmfh_dn.shape[1]+1] += U_LR_sum[:,0]
        
        # Notice that numpy.eigh() returns sorted eigenvalues and eigenvectors!
        tic = time.time()
        E_up, V_up = eigh(Hmfh_up)
        E_dn, V_dn = eigh(Hmfh_dn)       
        toc = time.time()
        logger.info('Iter %i: eigh done. (%.3f s)\n' %(i, toc-tic))
         
        # Calculate the density matrices
        rho_up = V_up[:,:n_up] @ V_up[:,:n_up].conj().T
        rho_dn = V_dn[:,:n_dn] @ V_dn[:,:n_dn].conj().T   
        
        # Deepcopy the arrays since np.diagonal() returns a read-only view
        n_ave_up_old = deepcopy(n_ave_up)
        n_ave_dn_old = deepcopy(n_ave_dn)
        n_ave_up_new = deepcopy(rho_up.diagonal())
        n_ave_dn_new = deepcopy(rho_dn.diagonal())    
        
        # Calculate the energy
        E_total_old = deepcopy(E_total)
        E_total = np.trace((rho_up + rho_dn) @ Htb) \
            + U * np.sum((n_ave_up_new - 0.5) * (n_ave_dn_new - 0.5))
        if U_LR is not None:
            n_ave = (n_ave_up_new + n_ave_dn_new - 1)[:,None]
            E_total += 0.5 * n_ave.conj().T @ U_LR @ n_ave
        
        delta_E = np.absolute(E_total - E_total_old)

        logger.info('E_total = %.15f eV\n' %(E_total) \
            + 'delta_E = %.15f (%.1e)\n\n' %(delta_E, delta_E))
        
        if delta_E < delta_E_lim:
            break
        else:
            # Calculate the electron densities for next iteration
            n_ave_up = mix_ratio * n_ave_up_new + mix_ratio_old * n_ave_up_old
            n_ave_dn = mix_ratio * n_ave_dn_new + mix_ratio_old * n_ave_dn_old
    
    # Print info
    toc_mfh = time.time()
    logger.info('Total number of iterations = %i\n' %(i + 1) \
        + 'mfh done. (%.3f s)\n\n' %(toc_mfh - tic_mfh))
         
    return Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn, E_total
    

def init_mfh_density(
    initial_density, Htb, n_site, n_up, n_dn, ind_up, ind_dn
):
    # tight-binding
    if initial_density == 0:
        E, V = eigh(Htb)
        rho_up = V[:,:n_up] @ V[:,:n_up].conj().T
        rho_dn = V[:,:n_dn] @ V[:,:n_dn].conj().T 
        n_ave_up = deepcopy(rho_up.diagonal())
        n_ave_dn = deepcopy(rho_dn.diagonal())
        
    # tight-binding + spin symmetry breaking
    elif initial_density == 1:
        E, V = eigh(Htb)
        rho_up = V[:,:n_up] @ V[:,:n_up].conj().T
        rho_dn = V[:,:n_dn] @ V[:,:n_dn].conj().T 
        n_ave_up = deepcopy(rho_up.diagonal()) \
                   + (np.random.rand(n_site) - 0.5) * 2e-3 \
                   + (np.random.rand(n_site) - 0.5) * 2e-2
        n_ave_dn = deepcopy(rho_dn.diagonal()) \
                   + (np.random.rand(n_site) - 0.5) * 2e-3 \
                   + (np.random.rand(n_site) - 0.5) * 2e-2

    # spin_order
    elif initial_density == 2:
        n_ave_up = np.zeros(n_site)
        n_ave_dn = np.zeros(n_site)
        n_ave_up[ind_up] = 1.0
        n_ave_dn[ind_dn] = 1.0
        
    # random(integer)
    elif initial_density == 3:
        n_ave_up = np.zeros(n_site)
        n_ave_dn = np.zeros(n_site)
        ind = np.random.permutation(np.arange(n_site))
        n_ave_up[ind[:n_up]] = 1
        ind = np.random.permutation(np.arange(n_site))
        n_ave_dn[ind[:n_dn]] = 1 
        
    # random(float)
    # Since U^T * U = U * U^T = I,
    # square sum of any row of an orthonormal matrix should be equal to 1.
    # We are sampling 'n_up' ('n_dn') of the orthonormal vectors, 
    # so the sum always be less than 1.
    # This gives us desired random electron densities.
    # Moreover, if we sum them up once more, it equals to the inner product 
    # of 'n_up' ('n_dn') vectors with themselves.
    # Thus, the sum will give 'n_up' ('n_dn') since they are orthonormal.
    elif initial_density == 4:
        Q = np.linalg.qr(
            np.random.rand(n_site,n_site),
            mode='complete'
        )[0]
        #rho_up = Q[:,:n_up] @ Q[:,:n_up].conj().T
        #n_ave_up = deepcopy(rho_up.diagonal())
        n_ave_up = (Q[:,:n_up]**2).sum(axis=1)
        
        Q = np.linalg.qr(
            np.random.rand(n_site,n_site),
            mode='complete'
        )[0]
        #rho_dn = Q[:,:n_dn] @ Q[:,:n_dn].conj().T
        #n_ave_dn = deepcopy(rho_dn.diagonal()) 
        n_ave_dn = (Q[:,:n_dn]**2).sum(axis=1)
        
    # zero
    elif initial_density == 5:
        n_ave_up = np.zeros(n_site)
        n_ave_dn = np.zeros(n_site)

    return n_ave_up, n_ave_dn


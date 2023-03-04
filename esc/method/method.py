"""
Wrapper module of methods

Usage:
method.start() : Starts the diagonalization depending on mode.
    For tb; diagonalizes the tight-binding Hamiltonian.
    For mfh; starts the self-consistent loop for mean-field Hubbard Hamiltonian.
method.orb_coef() : Returns the orbital coefficients

Diagonalization:
For eigenvalues and eigenvectors, numpy.eigh() function is used since the
Hamiltonians are real symmetric matrices. Otherwise, use numpy.eig() function.

Author: Gokhan Oztarhan
Created date: 10/12/2021
Last modified: 02/03/2023
"""

import logging

import numpy as np

from .. import config as cfg
from ..auconverter import AUConverter
from .hamiltonian import hamiltonian_tb
from .tb import tb
from .mfh import init_mfh_density, mfh


logger = logging.getLogger(__name__)

start = None
orb_coef = None
Htb = None
E = None
V = None
Hmfh_up = None
Hmfh_dn = None
E_up = None
E_dn = None
V_up = None
V_dn = None
E_total = None
E_total_nau = None


def init():
    global start, orb_coef
    global Htb, E, V
    global Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn
    global E_total, E_total_nau

    # Reset variables
    start = None
    orb_coef = None
    Htb = None
    E = None
    V = None
    Hmfh_up = None
    Hmfh_dn = None
    E_up = None
    E_dn = None
    V_up = None
    V_dn = None
    E_total = None
    E_total_nau = None

    # Print initialization info
    logger.info('[method]\n--------\n')

    # Form the tight-binding Hamiltonian matrix
    Htb = hamiltonian_tb(
        cfg.t, cfg.tp, cfg.ind_NN, cfg.ind_NN_2nd, cfg.n_site
    )

    # Set the module functions
    if cfg.mode == 'tb':
        start = _method_tb
        orb_coef = _orb_coef_tb

    elif cfg.mode == 'mfh':
        start = _method_mfh
        orb_coef = _orb_coef_mfh
            
            
def _method_tb():
    global E, V, E_total, E_total_nau
    
    # Diagonalization of the tight-binding Hamiltonian
    E, V, E_total = tb(Htb, cfg.n_up, cfg.n_dn)
    
    # Convert energy to given units
    auconverter = AUConverter(m_r=cfg.m_r, kappa=cfg.kappa)
    E_total_nau = auconverter.energy_to_SI(E_total, cfg.eunit)
    
    logger.info('E_total = %.15f\n' %E_total \
        + 'E_total_nau = %.15f %s\n\n' %(E_total_nau, cfg.eunit))


def _method_mfh():
    global Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn, E_total, E_total_nau
    
    # Initialize trial electron densities
    n_ave_up, n_ave_dn = init_mfh_density(
        cfg.initial_density, Htb, cfg.n_site, 
        cfg.n_up, cfg.n_dn, 
        cfg.ind_up, cfg.ind_dn
    )

    # Self-consistent loop for the mean-field Hubbard Hamiltonian
    Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn, E_total = mfh(
        Htb, cfg.U, cfg.mix_ratio, cfg.delta_E_lim, cfg.iter_lim, 
        cfg.n_up, cfg.n_dn, n_ave_up, n_ave_dn, U_LR=cfg.U_LR
    )
    
    # Convert energy to given units
    auconverter = AUConverter(m_r=cfg.m_r, kappa=cfg.kappa)
    E_total_nau = auconverter.energy_to_SI(E_total, cfg.eunit)
    
    logger.info('E_total = %.15f\n' %E_total \
        + 'E_total_nau = %.15f %s\n\n' %(E_total_nau, cfg.eunit))

            
def _orb_coef_tb():
    # rows: orbitals, columns: coefficients
    coef_up = V[:,:cfg.n_up].T
    coef_dn = V[:,:cfg.n_dn].T
    return np.vstack([coef_up, coef_dn])


def _orb_coef_mfh():
    # rows: orbitals, columns: coefficients
    coef_up = V_up[:,:cfg.n_up].T
    coef_dn = V_dn[:,:cfg.n_dn].T
    return np.vstack([coef_up, coef_dn])

     

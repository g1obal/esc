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
Last modified: 19/05/2024
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
overlap_eigstates = None
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
p_edge_pol = None


def init():
    global start, orb_coef, overlap_eigstates
    global Htb, E, V
    global Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn
    global E_total, E_total_nau
    global p_edge_pol

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
    p_edge_pol = None

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
        overlap_eigstates = _overlap_eigstates_tb

    elif cfg.mode == 'mfh':
        start = _method_mfh
        orb_coef = _orb_coef_mfh
        overlap_eigstates = _overlap_eigstates_mfh
            
            
def _method_tb():
    global E, V, E_total, E_total_nau, p_edge_pol
    
    # Diagonalization of the tight-binding Hamiltonian
    E, V, E_total = tb(Htb, cfg.n_up, cfg.n_dn)
    
    # Convert energy to given units
    auconverter = AUConverter(m_r=cfg.m_r, kappa=cfg.kappa)
    E_total_nau = auconverter.energy_to_SI(E_total, cfg.eunit)
    
    logger.info('E_total = %.15f\n' %E_total \
        + 'E_total_nau = %.15f %s\n\n' %(E_total_nau, cfg.eunit))
        
    # Edge polarization
    p_edge_pol = _edge_pol(V, V)
    logger.info('p_edge_pol = %.15f\n\n' %p_edge_pol)


def _method_mfh():
    global Hmfh_up, Hmfh_dn, E_up, E_dn, V_up, V_dn, E_total, E_total_nau
    global p_edge_pol
    
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
        
    # Edge polarization
    p_edge_pol = _edge_pol(V_up, V_dn)
    logger.info('p_edge_pol = %.15f\n\n' %p_edge_pol)


def _overlap_eigstates_tb():
    prod = V @ V.T
    ind = np.array(np.where(np.abs(prod) > cfg.overlap_threshold)).T
    overlap = np.hstack([ind, prod[ind[:,0],ind[:,1]][:,None]])
    
    return overlap, overlap
    
    
def _overlap_eigstates_mfh():
    prod = V_up @ V_up.T
    ind = np.array(np.where(np.abs(prod) > cfg.overlap_threshold)).T
    overlap_up = np.hstack([ind, prod[ind[:,0],ind[:,1]][:,None]])
    
    prod = V_dn @ V_dn.T
    ind = np.array(np.where(np.abs(prod) > cfg.overlap_threshold)).T
    overlap_dn = np.hstack([ind, prod[ind[:,0],ind[:,1]][:,None]])
    
    return overlap_up, overlap_dn


def _orb_coef_tb():
    # rows: orbitals, columns: coefficients
    coef_up = V.T
    coef_dn = V.T
    if cfg.full_orb_coef:
        return np.vstack([coef_up[:cfg.n_up,:], coef_dn[:cfg.n_dn,:]]), \
            coef_up, coef_dn
    else:
        return np.vstack([coef_up[:cfg.n_up,:], coef_dn[:cfg.n_dn,:]]), \
            None, None


def _orb_coef_mfh():
    # rows: orbitals, columns: coefficients
    coef_up = V_up.T
    coef_dn = V_dn.T
    if cfg.full_orb_coef:
        return np.vstack([coef_up[:cfg.n_up,:], coef_dn[:cfg.n_dn,:]]), \
            coef_up, coef_dn
    else:
        return np.vstack([coef_up[:cfg.n_up,:], coef_dn[:cfg.n_dn,:]]), \
            None, None


def _edge_pol(V_up, V_dn, include_corners=False):
    """
    Edge Polarization
    p = (<|s_edge|> - <|s_bulk|>) / (<|s_edge|> + <|s_bulk|>)
    where bulk is the interior lattice points.
    
    p =  1 : all total spin are polarized at the edges
    p =  0 : no polarization at the edges
    p = -1 : all total spin are polarized at the bulk
    """
    # Calculate spin densities
    probs = np.conj(V_up) * V_up # there is no transpose since this is psi^2
    n_ave_up = probs[:,:cfg.n_up].sum(axis=1)
    
    probs = np.conj(V_dn) * V_dn # there is no transpose since this is psi^2
    n_ave_dn = probs[:,:cfg.n_dn].sum(axis=1)
    
    spins = n_ave_up - n_ave_dn

    # Find edge and bulk indices
    ind, count = np.unique(cfg.ind_NN.flatten(), return_counts=True)
    ind_edge = ind[count < 3]
    ind_bulk = ind[count > 2]
    
    # Drop corner sites from edge indices for triangular zigzag flake
    if not include_corners:
        dist = np.sqrt((cfg.pos[ind_edge,:]**2).sum(axis=1))
        ind_edge = ind_edge[dist < dist.max() - cfg.a / 8]
    
    # <|s_edge|>
    s_edge = np.abs(spins[ind_edge]).mean()

    # <|s_bulk|>
    s_bulk = np.abs(spins[ind_bulk]).mean()
    
    # Edge polarization
    p_edge_pol = (s_edge - s_bulk) / (s_edge + s_bulk)
    
    return p_edge_pol



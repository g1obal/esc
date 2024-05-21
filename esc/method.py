"""
Wrapper module of methods

Diagonalization:
For eigenvalues and eigenvectors, numpy.eigh() function is used since the
Hamiltonians are real symmetric matrices. Otherwise, use numpy.eig() function.

Author: Gokhan Oztarhan
Created date: 10/12/2021
Last modified: 21/05/2024
"""

import logging

import numpy as np

from .auconverter import AUConverter
from .hamiltonian import hamiltonian_tb
from .tb import tb
from .mfh import init_mfh_density, mfh


logger = logging.getLogger(__name__)


class Method():
    def __init__(self, cfg):
        # Print initialization info
        logger.info('[method]\n--------\n')
        
        self.mode = cfg.mode
        self.t = cfg.t
        self.tp = cfg.tp
        self.U = cfg.U
        if cfg.U_LR is None:
            self.U_LR = None
        else:
            self.U_LR = cfg.U_LR[:] # Create a view not a copy
                          # If the view is modified, original array is modified!
        self.ind_NN = cfg.ind_NN[:]
        self.ind_NN_2nd = cfg.ind_NN_2nd[:]
        self.n_site = cfg.n_site
        self.n_up = cfg.n_up
        self.n_dn = cfg.n_dn
        self.ind_up = cfg.ind_up[:]
        self.ind_dn = cfg.ind_dn[:]
        self.mix_ratio = cfg.mix_ratio
        self.delta_E_lim = cfg.delta_E_lim
        self.iter_lim = cfg.iter_lim
        self.initial_density = cfg.initial_density
        self.m_r = cfg.m_r
        self.kappa = cfg.kappa
        self.eunit = cfg.eunit
        self.overlap_threshold = cfg.overlap_threshold
        self.pos = cfg.pos[:]
        self.a = cfg.a
        
        # Form the tight-binding Hamiltonian matrix
        self.Htb = hamiltonian_tb(
            self.t, self.tp, self.ind_NN, self.ind_NN_2nd, self.n_site
        )

    def start(self):
        if self.mode == 'tb':
            # Diagonalization of the tight-binding Hamiltonian
            self.E, self.V, self.E_total = tb(self.Htb, self.n_up, self.n_dn)
            
            self.E_up , self.E_dn = self.E[:], self.E[:]
            self.V_up, self.V_dn = self.V[:], self.V[:]
        
        elif self.mode =='mfh':
            # Initialize trial electron densities
            n_ave_up, n_ave_dn = init_mfh_density(
                self.initial_density, self.Htb, self.n_site, 
                self.n_up, self.n_dn, 
                self.ind_up, self.ind_dn
            )

            # Self-consistent loop for the mean-field Hubbard Hamiltonian
            self.Hmfh_up, self.Hmfh_dn, \
            self.E_up, self.E_dn, \
            self.V_up, self.V_dn, self.E_total = mfh(
                self.Htb, self.U, 
                self.mix_ratio, self.delta_E_lim, self.iter_lim, 
                self.n_up, self.n_dn, n_ave_up, n_ave_dn, U_LR=self.U_LR
            )
            
        # Convert energy to given units
        auconverter = AUConverter(m_r=self.m_r, kappa=self.kappa)
        self.E_total_nau = auconverter.energy_to_SI(self.E_total, self.eunit)
        
        logger.info('E_total = %.15f\n' %self.E_total \
            + 'E_total_nau = %.15f %s\n\n' %(self.E_total_nau, self.eunit))
            
        # Edge polarization
        self.edge_pol()
        logger.info('p_edge_pol = %.15f\n\n' %self.p_edge_pol)

    def overlap_eigstates(self):
        prod = self.V_up @ self.V_up.T
        ind = np.array(np.where(np.abs(prod) > self.overlap_threshold)).T
        overlap_up = np.hstack([ind, prod[ind[:,0],ind[:,1]][:,None]])
        
        prod = self.V_dn @ self.V_dn.T
        ind = np.array(np.where(np.abs(prod) > self.overlap_threshold)).T
        overlap_dn = np.hstack([ind, prod[ind[:,0],ind[:,1]][:,None]])
        
        return overlap_up, overlap_dn

    def orb_coef(self, full_orb_coef=False):
        # rows: orbitals, columns: coefficients
        coef_up = self.V_up.T
        coef_dn = self.V_dn.T
        
        if full_orb_coef:
            return np.vstack([coef_up[:self.n_up,:], coef_dn[:self.n_dn,:]]), \
                coef_up, coef_dn
        else:
            return np.vstack([coef_up[:self.n_up,:], coef_dn[:self.n_dn,:]]), \
                None, None

    def edge_pol(self, include_corners=False):
        """
        Edge Polarization
        p = (<|s_edge|> - <|s_bulk|>) / (<|s_edge|> + <|s_bulk|>)
        where bulk is the interior lattice points.
        
        p =  1 : all total spin are polarized at the edges
        p =  0 : no polarization at the edges
        p = -1 : all total spin are polarized at the bulk
        """
        # Calculate spin densities
        # There is no transpose since this is psi^2
        probs = np.conj(self.V_up) * self.V_up
        n_ave_up = probs[:,:self.n_up].sum(axis=1)
        
        probs = np.conj(self.V_dn) * self.V_dn
        n_ave_dn = probs[:,:self.n_dn].sum(axis=1)
        
        spins = n_ave_up - n_ave_dn

        # Find edge and bulk indices
        ind, count = np.unique(self.ind_NN.flatten(), return_counts=True)
        ind_edge = ind[count < 3]
        ind_bulk = ind[count > 2]
        
        # Drop corner sites from edge indices for triangular zigzag flake
        if not include_corners:
            dist = np.sqrt((self.pos[ind_edge,:]**2).sum(axis=1))
            ind_edge = ind_edge[dist < dist.max() - self.a / 8]
        
        # <|s_edge|>
        s_edge = np.abs(spins[ind_edge]).mean()

        # <|s_bulk|>
        s_bulk = np.abs(spins[ind_bulk]).mean()
        
        # Edge polarization
        self.p_edge_pol = (s_edge - s_bulk) / (s_edge + s_bulk)



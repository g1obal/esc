"""
Data module

Author: Gokhan Oztarhan
Created date: 11/12/2021
Last modified: 02/04/2024
"""

import os
import sys
import logging
import time

import numpy as np

from . import config as cfg
from .method import method


logger = logging.getLogger(__name__)


def save_data():
    tic = time.time()
    
    # Print info header
    logger.info('[data]\n------\n')

    # Save data
    if cfg.mode == 'tb':
        data_dict = {
            'mode': cfg.mode,
            'm_r': cfg.m_r,
            'kappa': cfg.kappa,
            'eunit': cfg.eunit,
            'lunit': cfg.lunit,
            'a': cfg.a,
            'a_nau': cfg.a_nau,
            'pos': cfg.pos,
            'ind_NN': cfg.ind_NN,
            'ind_NN_2nd': cfg.ind_NN_2nd,
            't': cfg.t,
            'tp': cfg.tp,
            't_nau': cfg.t_nau,
            'tp_nau': cfg.tp_nau,
            'total_charge': cfg.total_charge,
            'n_elec': cfg.n_elec,
            'n_up': cfg.n_up, 
            'n_dn': cfg.n_dn,
            'Htb': method.Htb,
            'E': method.E,
            'V': method.V,
            'E_total': method.E_total,
            'E_total_nau': method.E_total_nau,
            'p_edge_pol': method.p_edge_pol,
        }
    elif cfg.mode == 'mfh':
        data_dict = {
            'mode': cfg.mode,
            'm_r': cfg.m_r,
            'kappa': cfg.kappa,
            'eunit': cfg.eunit,
            'lunit': cfg.lunit,
            'a': cfg.a,
            'a_nau': cfg.a_nau,
            'pos': cfg.pos,
            'ind_NN': cfg.ind_NN,
            'ind_NN_2nd': cfg.ind_NN_2nd,
            't': cfg.t,
            'tp': cfg.tp,
            't_nau': cfg.t_nau,
            'tp_nau': cfg.tp_nau,
            'U': cfg.U,
            'U_nau': cfg.U_nau,
            'U_long_range': cfg.U_long_range,
            'U1_U2_scaling': cfg.U1_U2_scaling,
            'U1': cfg.U1,
            'U2': cfg.U2,
            'U3': cfg.U3,
            'U1_nau': cfg.U1_nau,
            'U2_nau': cfg.U2_nau,
            'U3_nau': cfg.U3_nau,
            'mix_ratio': cfg.mix_ratio,
            'delta_E_lim': cfg.delta_E_lim,
            'iter_lim': cfg.iter_lim,
            'initial_density': cfg.initial_density,
            'random_seed': cfg.random_seed,
            'total_charge': cfg.total_charge,
            'n_elec': cfg.n_elec,
            'n_up': cfg.n_up,
            'n_dn': cfg.n_dn,
            'Hmfh_up': method.Hmfh_up,
            'Hmfh_dn': method.Hmfh_dn,
            'E_up': method.E_up,
            'E_dn': method.E_dn,
            'V_up': method.V_up,
            'V_dn': method.V_dn,
            'E_total': method.E_total,
            'E_total_nau': method.E_total_nau,
            'p_edge_pol': method.p_edge_pol,
        }
  
    fname = os.path.join(cfg.root_dir, cfg.data_file)      
    
    # Here, the data_dict dictionary is not dumped into npz file,
    # instead it is given as keyword argument to numpy savez_compressed 
    # function. Therefore, numpy arrays are dumped separately.
    # If dictionary is dumped into savez_compressed function, numpy
    # calls pickle to serialize the dictionary object, which is not 
    # memory and cpu-time efficient.         
    np.savez_compressed(fname, **data_dict)
    logger.info('Data saved: %s\n' %fname)
    
    # Write orbital coefficients
    if cfg.verbose_orb_coef:  
        fname = os.path.join(cfg.root_dir, 'orb_dot_coef')
        np.savetxt(
            fname, method.orb_coef(), 
            fmt='% .12f', delimiter=' ', newline='\n'
        )
        logger.info('Orbital coefficients generated: %s\n' %fname)
    
    toc = time.time()
    logger.info('save_data done. (%.3f s)\n\n' %(toc - tic))

                          
def reload_data():
    tic = time.time()
    
    # Print info header
    logger.info('[data]\n------\n')
    
    fname = os.path.join(cfg.root_dir, cfg.data_file)
    
    if os.path.exists(fname):
        data = np.load(fname)
        
        toc = time.time()
        logger.info('reload_data done. (%.3f s)\n\n' %(toc - tic)) 
        return data
    else:
        sys.exit('Data file is not found: %s' %fname)



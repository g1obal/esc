"""
Config

Author: Gokhan Oztarhan
Created date: 06/03/2021
Last modified: 21/05/2024
"""

from copy import deepcopy
from os import urandom
import logging

import numpy as np

from .auconverter import AUConverter
from .latgen import Lattice


logger = logging.getLogger(__name__)


def _type_or_None(_type, variable):
    return [variable if variable is None else _type(variable)][0]


class Config():
    _FLK_TYPE = {
        0: 'hexagonal_zigzag',
        1: 'hexagonal_armchair',
        2: 'triangular_zigzag',
        3: 'triangular_armchair',
        4: 'nanoribbon'
    }               
    _INITIAL_DENSITY = {
        0: 'tight-binding',
        1: 'tight-binding + spin symmetry breaking',
        2: 'spin_order',
        3: 'random(integer)',
        4: 'random(float)',
        5: 'zero'
    }

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        self.prep()
        
    def prep(self):
        self.verbose_file = bool(self.verbose_file)
        self.verbose_console = bool(self.verbose_console)
        
        self.root_dir = str(self.root_dir)
        self.log_file = str(self.log_file)
        self.data_file = str(self.data_file)
        
        self.mode = str(self.mode) # 'tb': tight-binding, 
                                   # 'mfh': mean-field Hubbard
        
        self.random_seed = _type_or_None(int, self.random_seed) # random seed
        
        self.m_r = float(self.m_r) # effective electron mass ratio, m_eff / m_e
        self.kappa = float(self.kappa) # dielectric constant, 
                                       # varepsilon / varepsilon_0
        self.eunit = str(self.eunit) # energy units
        self.lunit = str(self.lunit) # length units
        
        self.t = float(self.t)
        self.tp = float(self.tp) # t prime is for 2nd nearest neighbor hopping
        
        self.U = float(self.U)
        # U1, U2, U3: 1st, 2nd and beyond 2nd nearest neighbor Coulomb int.
        #             set all to 0 (zero) to disable long range interactions
        #             None for calculation using 1/|r-r'|
        self.U1 = _type_or_None(float, self.U1)
        self.U2 = _type_or_None(float, self.U2)
        self.U3 = _type_or_None(float, self.U3)
        self.U1_U2_scaling = bool(self.U1_U2_scaling) # calculate U1 and/or U2 
                                            # from U using graphene parameters
        self.mix_ratio = float(self.mix_ratio) # new density proportion
        self.delta_E_lim = float(self.delta_E_lim) # energy difference threshold
                                                   # to end self consistent loop
        self.iter_lim = int(self.iter_lim) # iteration limit
        self.initial_density = int(self.initial_density)
                                # 0: tight-binding,
                                # 1: tight-binding + spin symmetry breaking,
                                # 2: spin_order,
                                # 3: random(integer), 
                                # 4: random(float),
                                # 5: zero
        
        self.total_charge = _type_or_None(int, self.total_charge) 
                            # set total number of electrons, n_elec
                            # None or 0 for charge neutral system
        self.Sz = _type_or_None(int, self.Sz) 
                  # total spin; to calculate the number of up and down electrons
                  # None to arrange n_up and n_dn according to Lieb's theorem
        self.spin_order = str(self.spin_order) 
                          # electrons are located in a spin order (for MFH init)
                          # AFM: antiferromagnetic, FM: ferromagnetic
        self.spin_order_direction = int(self.spin_order_direction) 
                               # the direction in which electrons are located
                               # 0: add electrons from outside to inside, 
                               #    additional electrons from inside to outside.
                               # 1: add electrons from inside to outside, 
                               #    additional electrons from outside to inside.

        self.a = float(self.a)
        self.n_side = int(self.n_side)
        self.width = int(self.width) # for nanoribbon
        self.bc = str(self.bc) # for nanoribbon
        self.lat_type = str(self.lat_type)
        self.flk_type = int(self.flk_type) # 0: hexagonal_zigzag, 
                                           # 1: hexagonal_armchair, 
                                           # 2: triangular_zigzag, 
                                           # 3: triangular_armchair, 
                                           # 4: nanoribbon
                     
        self.overlap_eigstates = bool(self.overlap_eigstates)
        self.overlap_threshold = float(self.overlap_threshold)

        self.orb_coef = bool(self.orb_coef)
        self.full_orb_coef = bool(self.full_orb_coef)
        
        self.plot = bool(self.plot)
        self.n_up_start = int(self.n_up_start)
        self.n_up_end = _type_or_None(int, self.n_up_end)
        self.n_dn_start = int(self.n_dn_start)
        self.n_dn_end = _type_or_None(int, self.n_dn_end)
        self.plot_E_limit = _type_or_None(float, self.plot_E_limit)
        self.dos_kde_sigma = _type_or_None(float, self.dos_kde_sigma)
        self.psi2_kde_sigma = _type_or_None(float, self.psi2_kde_sigma)
        self.mesh_resolution = int(self.mesh_resolution)    
        self.plot_fname = str(self.plot_fname)  
        self.plot_dpi = int(self.plot_dpi)
        self.plot_format = str(self.plot_format)

    def setup(self):
        # Copy input values to the variables with _nau suffix; 
        # nau means non-atomic-units (SI units or common units such as eV).
        # All quantities without _nau suffix will be in atomic units.
        self.t_nau = deepcopy(self.t)
        self.tp_nau = deepcopy(self.tp)
        self.U_nau = deepcopy(self.U)
        self.U1_nau = deepcopy(self.U1)
        self.U2_nau = deepcopy(self.U2)
        self.U3_nau = deepcopy(self.U3)
        self.a_nau = deepcopy(self.a)
        
        # Initialize atomic units converter
        auconverter = AUConverter(m_r=self.m_r, kappa=self.kappa)
        
        self.t = auconverter.energy_to_au(self.t_nau, self.eunit)
        self.tp = auconverter.energy_to_au(self.tp_nau, self.eunit)
        self.a = auconverter.length_to_au(self.a_nau, self.lunit)

        logger.info('[lattice]\n---------\n')
        
        # Shift lattice center of mass to origin
        if 'triangular' in self._FLK_TYPE[self.flk_type]:
            com_to_origin = True
        else:
            com_to_origin = False
        
        lattice = Lattice(
            self.lat_type, self._FLK_TYPE[self.flk_type], 
            self.a, self.n_side, width=self.width, bc=self.bc,
            com_to_origin=com_to_origin
        )
        
        lattice.set(
            total_charge=self.total_charge, 
            Sz=self.Sz, 
            spin_order=self.spin_order,
            spin_order_direction=self.spin_order_direction
        )

        self.n_site = lattice.n_site
        self.n_elec = lattice.n_elec
        self.n_up = lattice.n_up
        self.n_dn = lattice.n_dn
        self.ind_up = lattice.ind_up
        self.ind_dn = lattice.ind_dn
        
        self.pos = lattice.pos
        self.ind_NN = lattice.ind_NN
        self.ind_NN_2nd = lattice.ind_NN_2nd
                
        self.Sz_calc = (self.n_up - self.n_dn) * 0.5
        
        if self.mode == 'mfh':
            self.U = auconverter.energy_to_au(self.U_nau, self.eunit)

            if self.U1_nau == 0 and self.U2_nau == 0 and self.U3_nau == 0:
                self.U_long_range = False
                self.U_LR = None
            else:
                self.U_long_range = True
                
                # Long range interaction matrix in atomic units
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.U_LR = 1 / lattice.distances

                # 1st nearest neighbor Coulomb interactions
                if self.U1_nau is None:
                    if self.U1_U2_scaling:
                        self.U1_nau = self.U_nau / 1.9123
                        self.U1 = self.U / 1.9123
                else:
                    self.U1 = auconverter.energy_to_au(self.U1_nau, self.eunit)

                # 2nd nearest neighbor Coulomb interactions
                if self.U2_nau is None:
                    if self.U1_U2_scaling:
                        self.U2_nau = self.U_nau / 3.098068629
                        self.U2 = self.U / 3.098068629
                else:
                    self.U2 = auconverter.energy_to_au(self.U2_nau, self.eunit)
                
                # Beyond 2nd nearest neighbor Coulomb interactions
                if self.U3_nau is not None:
                    self.U3 = auconverter.energy_to_au(self.U3_nau, self.eunit)
                    self.U_LR[:,:] = self.U3 # it should be assigned 
                                             # before U1 and U2
                    
                # Set values in long range interaction matrix
                if self.U1_nau is not None:
                    self.U_LR[self.ind_NN[:,0],self.ind_NN[:,1]] = self.U1
                    self.U_LR[self.ind_NN[:,1],self.ind_NN[:,0]] = self.U1
                if self.U2_nau is not None:
                    self.U_LR[self.ind_NN_2nd[:,0],self.ind_NN_2nd[:,1]] = \
                        self.U2
                    self.U_LR[self.ind_NN_2nd[:,1],self.ind_NN_2nd[:,0]] = \
                        self.U2
                
                # Diagonal elements should be zero, since on-site interactions
                # are included in Hmfh_up and Hmfh_dn matrices in mfh.py.
                self.U_LR.ravel()[::self.U_LR.shape[1]+1] = 0e0
        else:
            self.U_long_range = None
            self.U_LR = None
                
        # Set the numpy global random state. This affects all numpy.random 
        # calls throughout the modules in which numpy is imported. We set
        # the global random state since this program is not expected to be
        # imported by another python script or module. Otherwise, use
        # numpy.random.RandomState, and pass it to the desired function.
        if self.random_seed is None:
            # os.urandom(4) generates a bytestring of 32 bits 
            # from unpredictable OS dependent sources. 
            self.random_seed = int.from_bytes(urandom(4), 'big', signed=False)
            self.random_seed_auto_set = True
        else:
            self.random_seed_auto_set = False
        np.random.seed(self.random_seed)

        # Set default values of n_up_end and n_dn_end
        if self.n_up_end is None:
            self.n_up_end = self.n_up
        if self.n_dn_end is None:
            self.n_dn_end = self.n_dn

    def print_info(self):
        logger.info('\n[config]\n--------\n')
        
        random_seed_line = 'random_seed = %i' %self.random_seed
        if self.random_seed_auto_set:
            random_seed_line += ' (obtained from os.urandom)'
        
        if self.mode == 'tb':
            string = 'mode = %s\n\n' %self.mode \
                   + random_seed_line + '\n\n' \
                   + 'm_r = %.5e\n' %self.m_r \
                   + 'kappa = %.5e\n\n' %self.kappa \
                   + 't = %.5e, t_nau = %.5e %s\n' \
                    %(self.t, self.t_nau, self.eunit) \
                   + 'tp = %.5e, tp_nau = %.5e %s\n\n' \
                    %(self.tp, self.tp_nau, self.eunit) \
                   + 'a = %.5e, a_nau = %.5e %s\n\n' \
                    %(self.a, self.a_nau, self.lunit) \
                   + 'total_charge = %s\n' %self.total_charge \
                   + 'Sz = %-6s\n' %self.Sz \
                   + 'spin_order = %s\n' %self.spin_order \
                   + 'spin_order_direction = %s\n\n' \
                    %self.spin_order_direction \
                   + 'n_elec = %i\n' %self.n_elec \
                   + 'n_up = %i\n' %self.n_up \
                   + 'n_dn = %i\n' %self.n_dn \
                   + 'Sz_calc = %.1f\n\n' %self.Sz_calc
                   
        elif self.mode == 'mfh':
            if self.U1 is None:
                U1_line = 'U1 = %s, U1_nau = %s' %(self.U1, self.U1_nau)
            else:
                U1_line = 'U1 = %.5e, U1_nau = %.5e %s' \
                    %(self.U1, self.U1_nau, self.eunit)
            if self.U2 is None:
                U2_line = 'U2 = %s, U2_nau = %s' %(self.U2, self.U2_nau)
            else:
                U2_line = 'U2 = %.5e, U2_nau = %.5e %s' \
                    %(self.U2, self.U2_nau, self.eunit)
            if self.U3 is None:
                U3_line = 'U3 = %s, U3_nau = %s' %(self.U3, self.U3_nau)
            else:
                U3_line = 'U3 = %.5e, U3_nau = %.5e %s' \
                    %(self.U3, self.U3_nau, self.eunit)

            if self.U_LR is None:
                U_LR_line = 'U_LR = None'
            else:
                U_LR_line = 'U_LR != None'
        
            string = 'mode = %s\n\n' %self.mode \
               + random_seed_line + '\n\n' \
               + 'm_r = %.5e\n' %self.m_r \
               + 'kappa = %.5e\n\n' %self.kappa \
               + 't = %.5e, t_nau = %.5e %s\n' \
                %(self.t, self.t_nau, self.eunit) \
               + 'tp = %.5e, tp_nau = %.5e %s\n\n' \
                %(self.tp, self.tp_nau, self.eunit) \
               + 'U = %.5e, U_nau = %.5e %s\n' \
                %(self.U, self.U_nau, self.eunit) \
               + 'U/t = %.5f\n' %(self.U / self.t) \
               + 'U_nau/t_nau = %.5f\n\n' %(self.U_nau / self.t_nau) \
               + 'U_long_range = %s\n' %self.U_long_range \
               + U_LR_line + '\n' \
               + 'U1_U2_scaling = %s\n' %self.U1_U2_scaling \
               + U1_line + '\n' \
               + U2_line + '\n' \
               + U3_line + '\n\n' \
               + 'mix_ratio = %.3f\n' %self.mix_ratio \
               + 'delta_E_lim = %.1e\n' %self.delta_E_lim \
               + 'iter_lim = %i\n' %self.iter_lim \
               + 'initial_density = %d (%s)\n\n' \
                %(self.initial_density, 
                self._INITIAL_DENSITY[self.initial_density]) \
               + 'a = %.5e, a_nau = %.5e %s\n\n' \
                %(self.a, self.a_nau, self.lunit) \
               + 'total_charge = %s\n' %self.total_charge \
               + 'Sz = %-6s\n' %self.Sz \
               + 'spin_order = %s\n' %self.spin_order \
               + 'spin_order_direction = %s\n\n' %self.spin_order_direction \
               + 'n_elec = %i\n' %self.n_elec \
               + 'n_up = %i\n' %self.n_up \
               + 'n_dn = %i\n' %self.n_dn \
               + 'Sz_calc = %.1f\n\n' %self.Sz_calc
               
        logger.info(string)



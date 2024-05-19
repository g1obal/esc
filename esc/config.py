"""
Config Module

Defines global variables shared across the program.
Since python modules are imported "only once",
this module works similar to a singleton.

Author: Gokhan Oztarhan
Created date: 06/03/2021
Last modified: 19/05/2024
"""

from copy import deepcopy
from os import urandom
import logging

import numpy as np

from .configparserx import ConfigParserX
from .auconverter import AUConverter
from .latgen import Lattice


logger = logging.getLogger(__name__)

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

# [verbose]
verbose_file = 1 
verbose_console = 1

# [file]
root_dir = 'outputs'
log_file = 'output'
data_file = 'data.npz'

# [mode]
mode = 'tb' # 'tb': tight-binding, 'mfh': mean-field Hubbard

# [units]
m_r = 0.067 # effective electron mass ratio, m_eff / m_e
kappa = 12.4 # dielectric constant, varepsilon / varepsilon_0
eunit = 'meV' # energy units
lunit = 'nm' # length units

# [tb]
t = 1.0
tp = 0 # t prime is for 2nd nearest neighbor hopping

# [mfh]
U = 2.0
# U1, U2, U3: 1st, 2nd and beyond 2nd nearest neighbor Coulomb interactions
#             set all to 0 (zero) to disable long range interactions
#             None for calculation using 1/|r-r'|
U1 = 0
U2 = 0
U3 = 0
U1_U2_scaling = False # calculate U1 and/or U2 from U using graphene parameters
                      # if they are set to None.
mix_ratio = 0.7 # new density proportion
delta_E_lim = 1e-13 # energy difference threshold to end self consistent loop
iter_lim = 1000 # iteration limit
initial_density = 2 # 0: tight-binding,
                    # 1: tight-binding + spin symmetry breaking,
                    # 2: spin_order,
                    # 3: random(integer), 
                    # 4: random(float),
                    # 5: zero
random_seed = None # random seed for initial density

# [electron]
total_charge = None # set total number of electrons, n_elec
                    # None or 0 for charge neutral system
Sz = None # total spin; to calculate the number of up and down electrons
          # None to arrange n_up and n_dn according to Lieb's theorem
spin_order = 'AFM' # electrons are located in a spin order (for MFH init)
                   # AFM: antiferromagnetic, FM: ferromagnetic
spin_order_direction = 1 # the direction in which electrons are located
                         # 0: add electrons from outside to inside, 
                         #    additional electrons from inside to outside.
                         # 1: add electrons from inside to outside, 
                         #    additional electrons from outside to inside. 

# [lattice]
a = 50
n_side = 4
width = 1 # for nanoribbon
bc = 'xy' # for nanoribbon
lat_type = 'honeycomb'
flk_type = 1 # 0: hexagonal_zigzag, 
             # 1: hexagonal_armchair, 
             # 2: triangular_zigzag, 
             # 3: triangular_armchair, 
             # 4: nanoribbon

# [orb_coef]
orb_coef = 1
full_orb_coef = 0

# [plotting]
plot = 1
n_up_start = 0
n_up_end = None
n_dn_start = 0
n_dn_end = None
plot_E_limit = None
dos_kde_sigma = None
psi2_kde_sigma = None
mesh_resolution = 500    
plot_fname = 'summary'    
plot_dpi = 600
plot_format = 'jpg'

# Dynamic run variables
t_nau = None
tp_nau = None
U_nau = None
U1_nau = None
U2_nau = None
U3_nau = None
a_nau = None
U_long_range = None
U_LR = None
n_site = None
n_elec = None
n_up = None
n_dn = None
ind_up = None
ind_dn = None
pos = None
ind_NN = None
ind_NN_2nd = None
Sz_calc = None
random_seed_auto_set = False


def update(var_dict):
    """Update global variables using var_dict"""
    globals().update(var_dict)


def set():
    global t, tp, U, U1, U2, U3, a
    global t_nau, tp_nau, U_nau, U1_nau, U2_nau, U3_nau, a_nau
    global U_long_range, U_LR
    global n_site, n_elec, n_up, n_dn, ind_up, ind_dn
    global pos, ind_NN, ind_NN_2nd, Sz_calc
    global random_seed, random_seed_auto_set
    global n_up_start, n_up_end, n_dn_start, n_dn_end
    
    # Copy input values to the variables with _nau suffix; 
    # nau means non-atomic-units (SI units or common units such as eV).
    # All quantities without _nau suffix will be in atomic units.
    t_nau = deepcopy(t)
    tp_nau = deepcopy(tp)
    U_nau = deepcopy(U)
    U1_nau = deepcopy(U1)
    U2_nau = deepcopy(U2)
    U3_nau = deepcopy(U3)
    a_nau = deepcopy(a)
    
    # Initialize atomic units converter
    auconverter = AUConverter(m_r=m_r, kappa=kappa)
    
    t = auconverter.energy_to_au(t_nau, eunit)
    tp = auconverter.energy_to_au(tp_nau, eunit)
    a = auconverter.length_to_au(a_nau, lunit)

    logger.info('[lattice]\n---------\n')
    
    # Shift lattice center of mass to origin
    if 'triangular' in _FLK_TYPE[flk_type]:
        com_to_origin = True
    else:
        com_to_origin = False
    
    lattice = Lattice(
        lat_type, _FLK_TYPE[flk_type], 
        a, n_side, width=width, bc=bc,
        com_to_origin=com_to_origin
    )
    
    lattice.set(
        total_charge=total_charge, 
        Sz=Sz, 
        spin_order=spin_order,
        spin_order_direction=spin_order_direction
    )

    n_site = lattice.n_site
    n_elec = lattice.n_elec
    n_up = lattice.n_up
    n_dn = lattice.n_dn
    ind_up = lattice.ind_up
    ind_dn = lattice.ind_dn
    
    pos = lattice.pos
    ind_NN = lattice.ind_NN
    ind_NN_2nd = lattice.ind_NN_2nd
            
    Sz_calc = (n_up - n_dn) * 0.5
    
    if mode == 'mfh':
        U = auconverter.energy_to_au(U_nau, eunit)

        if U1_nau == 0 and U2_nau == 0 and U3_nau == 0:
            U_long_range = False
            U_LR = None
        else:
            U_long_range = True
            
            # Long range interaction matrix in atomic units
            with np.errstate(divide='ignore', invalid='ignore'):
                U_LR = 1 / lattice.distances

            # 1st nearest neighbor Coulomb interactions
            if U1_nau is None:
                if U1_U2_scaling:
                    U1_nau = U_nau / 1.9123
                    U1 = U / 1.9123
            else:
                U1 = auconverter.energy_to_au(U1_nau, eunit)

            # 2nd nearest neighbor Coulomb interactions
            if U2_nau is None:
                if U1_U2_scaling:
                    U2_nau = U_nau / 3.098068629
                    U2 = U / 3.098068629
            else:
                U2 = auconverter.energy_to_au(U2_nau, eunit)
            
            # Beyond 2nd nearest neighbor Coulomb interactions
            if U3_nau is not None:
                U3 = auconverter.energy_to_au(U3_nau, eunit)
                U_LR[:,:] = U3 # it should be assigned before U1 and U2
                
            # Set values in long range interaction matrix
            if U1_nau is not None:
                U_LR[ind_NN[:,0],ind_NN[:,1]] = U1
                U_LR[ind_NN[:,1],ind_NN[:,0]] = U1
            if U2_nau is not None:
                U_LR[ind_NN_2nd[:,0],ind_NN_2nd[:,1]] = U2
                U_LR[ind_NN_2nd[:,1],ind_NN_2nd[:,0]] = U2
            
            # Diagonal elements should be zero, since on-site interactions
            # are included in Hmfh_up and Hmfh_dn matrices in mfh.py.
            U_LR.ravel()[::U_LR.shape[1]+1] = 0e0
            
        # Set the numpy global random state for MFH initial density. This 
        # affects all numpy.random calls throughout the modules in which numpy
        # is imported. We are setting the global random state since this 
        # program is not expected to be imported by another python script or
        # module. Otherwise, use numpy.random.RandomState instead of
        # numpy.random.seed, and pass it to the desired function as argument.
        if random_seed is None:
            # os.urandom(4) generates a bytestring of 32 bits 
            # from unpredictable OS dependent sources. 
            random_seed = int.from_bytes(urandom(4), 'big', signed=False)
            random_seed_auto_set = True
        else:
            random_seed_auto_set = False
        np.random.seed(random_seed)

    # Set default values of n_up_end and n_dn_end
    if n_up_end is None:
        n_up_end = n_up
    if n_dn_end is None:
        n_dn_end = n_dn

       
def print_info():
    logger.info('\n[config]\n--------\n')
    
    if mode == 'tb':
        string = 'mode = %s\n\n' %mode \
               + 'm_r = %.5e\n' %m_r \
               + 'kappa = %.5e\n\n' %kappa \
               + 't = %.5e, t_nau = %.5e %s\n' %(t, t_nau, eunit) \
               + 'tp = %.5e, tp_nau = %.5e %s\n\n' %(tp, tp_nau, eunit) \
               + 'a = %.5e, a_nau = %.5e %s\n\n' %(a, a_nau, lunit) \
               + 'total_charge = %s\n' %total_charge \
               + 'Sz = %-6s\n' %Sz \
               + 'spin_order = %s\n' %spin_order \
               + 'spin_order_direction = %s\n\n' %spin_order_direction \
               + 'n_elec = %i\n' %n_elec \
               + 'n_up = %i\n' %n_up \
               + 'n_dn = %i\n' %n_dn \
               + 'Sz_calc = %.1f\n\n' %Sz_calc
               
    elif mode == 'mfh':
        if U1 is None:
            U1_line = 'U1 = %s, U1_nau = %s' %(U1, U1_nau)
        else:
            U1_line = 'U1 = %.5e, U1_nau = %.5e %s' %(U1, U1_nau, eunit)
        if U2 is None:
            U2_line = 'U2 = %s, U2_nau = %s' %(U2, U2_nau)
        else:
            U2_line = 'U2 = %.5e, U2_nau = %.5e %s' %(U2, U2_nau, eunit)
        if U3 is None:
            U3_line = 'U3 = %s, U3_nau = %s' %(U3, U3_nau)
        else:
            U3_line = 'U3 = %.5e, U3_nau = %.5e %s' %(U3, U3_nau, eunit)

        if U_LR is None:
            U_LR_line = 'U_LR = None'
        else:
            U_LR_line = 'U_LR != None'
        
        random_seed_line = 'random_seed = %i' %random_seed
        if random_seed_auto_set:
            random_seed_line += ' (obtained from os.urandom)'
    
        string = 'mode = %s\n\n' %mode \
           + 'm_r = %.5e\n' %m_r \
           + 'kappa = %.5e\n\n' %kappa \
           + 't = %.5e, t_nau = %.5e %s\n' %(t, t_nau, eunit) \
           + 'tp = %.5e, tp_nau = %.5e %s\n\n' %(tp, tp_nau, eunit) \
           + 'U = %.5e, U_nau = %.5e %s\n' %(U, U_nau, eunit) \
           + 'U/t = %.5f\n' %(U / t) \
           + 'U_nau/t_nau = %.5f\n\n' %(U_nau / t_nau) \
           + 'U_long_range = %s\n' %U_long_range \
           + U_LR_line + '\n' \
           + 'U1_U2_scaling = %s\n' %U1_U2_scaling \
           + U1_line + '\n' \
           + U2_line + '\n' \
           + U3_line + '\n\n' \
           + 'mix_ratio = %.3f\n' %mix_ratio \
           + 'delta_E_lim = %.1e\n' %delta_E_lim \
           + 'iter_lim = %i\n' %iter_lim \
           + 'initial_density = %d (%s)\n' \
           %(initial_density, _INITIAL_DENSITY[initial_density]) \
           + random_seed_line + '\n\n' \
           + 'a = %.5e, a_nau = %.5e %s\n\n' %(a, a_nau, lunit) \
           + 'total_charge = %s\n' %total_charge \
           + 'Sz = %-6s\n' %Sz \
           + 'spin_order = %s\n' %spin_order \
           + 'spin_order_direction = %s\n\n' %spin_order_direction \
           + 'n_elec = %i\n' %n_elec \
           + 'n_up = %i\n' %n_up \
           + 'n_dn = %i\n' %n_dn \
           + 'Sz_calc = %.1f\n\n' %Sz_calc
           
    logger.info(string)


def parse_config_file(fname):
    # Initialize ConfigParser
    parser = ConfigParserX()

    # Read config file
    parser.parse_file(fname)
    
    # [verbose]
    parser.set_type('verbose', 'verbose_file', bool)
    parser.set_type('verbose', 'verbose_console', bool)
    
    # [file]  
    parser.set_type('file', 'root_dir', str)
    parser.set_type('file', 'log_file', str)
    parser.set_type('file', 'data_file', str)
    
    # [mode]
    parser.set_type('mode', 'mode', str)
    
    # [units]
    parser.set_type('units', 'm_r', float)
    parser.set_type('units', 'kappa', float)
    parser.set_type('units', 'eunit', str)
    parser.set_type('units', 'lunit', str)
    
    # [tb]
    parser.set_type('tb', 't', float)
    parser.set_type('tb', 'tp', float)

    # [mfh]
    parser.set_type('mfh', 'U', float)
    parser.set_type('mfh', 'U1', float, can_be_None=True)
    parser.set_type('mfh', 'U2', float, can_be_None=True)
    parser.set_type('mfh', 'U3', float, can_be_None=True)
    parser.set_type('mfh', 'U1_U2_scaling', bool)
    parser.set_type('mfh', 'mix_ratio', float)
    parser.set_type('mfh', 'delta_E_lim', float)
    parser.set_type('mfh', 'iter_lim', int)
    parser.set_type('mfh', 'initial_density', int)
    parser.set_type('mfh', 'random_seed', int, can_be_None=True)

    # [electron]
    parser.set_type('electron', 'total_charge', int, can_be_None=True)
    parser.set_type('electron', 'Sz', float, can_be_None=True)
    parser.set_type('electron', 'spin_order', str)
    parser.set_type('electron', 'spin_order_direction', int)
        
    # [lattice]
    parser.set_type('lattice', 'a', float)
    parser.set_type('lattice', 'n_side', int)
    parser.set_type('lattice', 'width', int)
    parser.set_type('lattice', 'bc', str)
    parser.set_type('lattice', 'lat_type', str)
    parser.set_type('lattice', 'flk_type', int)
    
    # [orb_coef]
    parser.set_type('orb_coef', 'orb_coef', bool)
    parser.set_type('orb_coef', 'full_orb_coef', bool)
    
    # [plotting]
    parser.set_type('plotting', 'plot', bool)
    parser.set_type('plotting', 'n_up_start', int)
    parser.set_type('plotting', 'n_up_end', int, can_be_None=True)
    parser.set_type('plotting', 'n_dn_start', int)
    parser.set_type('plotting', 'n_dn_end', int, can_be_None=True)
    parser.set_type('plotting', 'plot_E_limit', float, can_be_None=True)
    parser.set_type('plotting', 'dos_kde_sigma', float, can_be_None=True)
    parser.set_type('plotting', 'psi2_kde_sigma', float, can_be_None=True)
    parser.set_type('plotting', 'mesh_resolution', int)
    parser.set_type('plotting', 'plot_fname', str)
    parser.set_type('plotting', 'plot_dpi', int)
    parser.set_type('plotting', 'plot_format', str)
   
    return parser.option_dict



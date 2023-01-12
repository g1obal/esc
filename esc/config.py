"""
Config Module

Defines global variables shared across the program.
Since python modules are imported "only once",
this module works similar to a singleton.

Author: Gokhan Oztarhan
Created date: 06/03/2021
Last modified: 11/01/2023
"""

from os import urandom
import logging
    
import numpy as np
    
from .configparserx import ConfigParserX
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
verbose_orb_coef = 1
verbose_plot = 1

# [file]
root_dir = 'outputs'
log_file = 'output'
data_file = 'data.npz'

# [mode]
mode = 'mfh' # 'tb': tight-binding, 'mfh': mean-field Hubbard

# [tb]
t = 1.0
tp = 0.0 # t prime is for 2nd nearest neighbor hopping

# [mfh]
U = 2.0
U_long_range = False
# U1, U2, U3: 1st, 2nd and beyond 2nd nearest neighbor Coulomb interactions, 
#             None for automatic calculation from U
#             U1, U2 and U3 are active when U_long_range is True
U1 = None
U2 = None 
U3 = None 
mix_ratio = 0.7 # new density proportion
delta_E_lim = 1e-11 # energy difference threshold to end self consistent loop
iter_lim = 500 # iteration limit
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
a = 50e-9
n_side = 4
width = 1 # for nanoribbon
bc = 'xy' # for nanoribbon
lat_type = 'honeycomb'
flk_type = 1 # 0: hexagonal_zigzag, 
             # 1: hexagonal_armchair, 
             # 2: triangular_zigzag, 
             # 3: triangular_armchair, 
             # 4: nanoribbon
         
# [plotting]
plot_E_limit = None
dos_kde_sigma = None
psi2_kde_sigma = None
mesh_resolution = 500        
plot_dpi = 600
plot_format = 'jpg'

# Dynamic run variables
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
U_LR = None
U1_auto_calc = False
U2_auto_calc = False
U3_auto_calc = False
random_seed_auto_set = False

                              
def update(var_dict):
    """Update global variables using var_dict"""
    globals().update(var_dict)
  

def set():
    global n_site, n_elec, n_up, n_dn, ind_up, ind_dn
    global pos, ind_NN, ind_NN_2nd, Sz_calc
    global U_LR, U1, U2, U3
    global U1_auto_calc, U2_auto_calc, U3_auto_calc
    global random_seed, random_seed_auto_set

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
        if U_long_range:
            # Beyond 2nd nearest neighbor Coulomb interactions
            if U3 is None:
                # 1 J = 6.24150974e+18 eV
                U_LR = (U / 16.522) \
                    * 6.24150974e+18 * (1.602176634e-19 * 1.602176634e-19) \
                    / (4 * np.pi * 8.8541878128e-12 * lattice.distances) 
                U3_auto_calc = True
            else:
                U_LR = np.full(lattice.distances.shape, U3)
                U3_auto_calc = False
                
            # 1st nearest neighbor Coulomb interactions
            if U1 is None:
                U1 = U / 1.9123
                U1_auto_calc = True
            else:
                U1_auto_calc = False
            U_LR[ind_NN[:,0],ind_NN[:,1]] = U1
            U_LR[ind_NN[:,1],ind_NN[:,0]] = U1
            
            # 2nd nearest neighbor Coulomb interactions
            if U2 is None:
                U2 = U / 3.098068629
                U2_auto_calc = True
            else:
                U2_auto_calc = False
            U_LR[ind_NN_2nd[:,0],ind_NN_2nd[:,1]] = U2
            U_LR[ind_NN_2nd[:,1],ind_NN_2nd[:,0]] = U2
            
            # Diagonal elements should be zero, since on-site interactions
            # are included in Hmfh_up and Hmfh_dn matrices in mfh.py.
            U_LR.ravel()[::U_LR.shape[1]+1] = 0e0
        else:
            U_LR = None
            
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
        
        
def print_info():
    logger.info('\n[config]\n--------\n')
    
    if mode == 'tb':
        string = 'mode = %s\n\n' %mode \
               + 't = %.5e eV\n' %t \
               + 'tp = %.5e eV\n\n' %tp \
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
            U1_line = 'U1 = %s' %U1
        else:
            U1_line = 'U1 = %.5e eV' %U1
        if U2 is None:
            U2_line = 'U2 = %s' %U2
        else:
            U2_line = 'U2 = %.5e eV' %U2
        if U3 is None:
            U3_line = 'U3 = %s' %U3
        else:
            U3_line = 'U3 = %.5e eV' %U3
        if U1_auto_calc:
            U1_line += ' (calculated from U)'
        if U2_auto_calc:
            U2_line += ' (calculated from U)'
        if U3_auto_calc:
            U3_line += ' (calculated from U)'
        
        random_seed_line = 'random_seed = %i' %random_seed
        if random_seed_auto_set:
            random_seed_line += ' (obtained from os.urandom)'
    
        string = 'mode = %s\n\n' %mode \
           + 't = %.5e eV\n' %t \
           + 'tp = %.5e eV\n\n' %tp \
           + 'U = %.5e eV\n' %U \
           + 'U_long_range = %s\n' %U_long_range \
           + U1_line + '\n' \
           + U2_line + '\n' \
           + U3_line + '\n\n' \
           + 'U/t = %.5f\n\n' %(U / t) \
           + 'mix_ratio = %.3f\n' %mix_ratio \
           + 'delta_E_lim = %.1e\n' %delta_E_lim \
           + 'iter_lim = %i\n' %iter_lim \
           + 'initial_density = %d (%s)\n' \
           %(initial_density, _INITIAL_DENSITY[initial_density]) \
           + random_seed_line + '\n\n' \
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
    parser.set_type('verbose', 'verbose_orb_coef', bool)
    parser.set_type('verbose', 'verbose_plot', bool)
    
    # [file]  
    parser.set_type('file', 'root_dir', str)
    parser.set_type('file', 'log_file', str)
    parser.set_type('file', 'data_file', str)
    
    # [mode]
    parser.set_type('mode', 'mode', str)
    
    # [tb]
    parser.set_type('tb', 't', float)
    parser.set_type('tb', 'tp', float)

    # [mfh]
    parser.set_type('mfh', 'U', float)
    parser.set_type('mfh', 'U_long_range', bool)
    parser.set_type('mfh', 'U1', float, can_be_None=True)
    parser.set_type('mfh', 'U2', float, can_be_None=True)
    parser.set_type('mfh', 'U3', float, can_be_None=True)
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
    
    # [plotting]
    parser.set_type('plotting', 'plot_E_limit', float, can_be_None=True)
    parser.set_type('plotting', 'dos_kde_sigma', float, can_be_None=True)
    parser.set_type('plotting', 'psi2_kde_sigma', float, can_be_None=True)
    parser.set_type('plotting', 'mesh_resolution', int)
    parser.set_type('plotting', 'plot_dpi', int)
    parser.set_type('plotting', 'plot_format', str)
   
    return parser.option_dict



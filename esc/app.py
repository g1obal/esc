"""
Electronic Structure Calculator

Author: Gokhan Oztarhan
Created date: 28/08/2019
Last modified: 03/12/2022
"""

import os
import time
import datetime
import logging

from . import config as cfg
from .method import method
from .logger import set_logger, unset_logger
from .data import save_data, reload_data
from .plotting import plot, replot


logger = logging.getLogger(__name__)


def run(config_dict):    
    """
    Main function of esc.
    Calls all necessary functions.
    """
    tic = time.time()
    
    # Update global variables of config
    cfg.update(config_dict)
    
    # Initialize root dir of all outputs
    if not os.path.exists(cfg.root_dir):
        os.mkdir(cfg.root_dir)

    # Initialize logger
    set_logger(
        cfg.verbose_file, cfg.verbose_console,
        filename=os.path.join(cfg.root_dir, cfg.log_file)
    )
    
    # Print start info
    logger.info('Electronic Structure Calculator\n' \
        + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\n\n'))

    # Set preferences
    cfg.set()
    
    # Print cfg info
    cfg.print_info()
    
    # Initialize and start method
    method.init()
    method.start()

    # Save data
    save_data() 

    # Plot figures
    if cfg.verbose_plot:
        plot()
     
    # Print end info
    toc = time.time()
    logger.info('\nExecution time = %.3f s\n' %(toc-tic))
    
    # Close all handlers of logger
    unset_logger()
    

def run_replot(config_dict):
    """Replot figures using existing data"""
    tic = time.time()
    
    # Update global variables of config
    cfg.update(config_dict)
    
    # Initialize logger for only printing to console
    set_logger(0, 1)
    
    # Print start info
    logger.info('Electronic Structure Calculator: Replot Figures\n\n')   
    
    # Reload data
    data = reload_data()
    
    # Replot figures
    replot(data)
    
    # Print end info
    toc = time.time()
    logger.info('\nExecution time = %.3f s\n' %(toc-tic))
    
    # Close all handlers of logger
    unset_logger()   
    


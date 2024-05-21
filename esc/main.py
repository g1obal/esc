"""
Electronic Structure Calculator

Main module

Author: Gokhan Oztarhan
Created date: 28/08/2019
Last modified: 21/05/2024
"""

import os
import time
import datetime
import logging
import argparse
import json

from .__init__ import __version__
from .config import Config
from .method import Method
from .logger import set_logger, unset_logger
from .data import save_data, reload_data
from .plotting import plot, replot


logger = logging.getLogger(__name__)


def main():
    # Parse command line arguments
    args = parse_args()

    # Parse config file
    with open(args.input_file, 'r') as f_input:
        config_dict = json.load(f_input)

    # Run esc or replot data
    if 'replot' in args:
        run_replot(config_dict)
    else:
        run(config_dict)


def run(config_dict):    
    """
    Main function of esc.
    Calls all necessary functions.
    """
    tic = time.time()
    
    # Initialize config
    cfg = Config(config_dict)
    
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

    # Set preferences and print information
    cfg.setup()
    cfg.print_info()
    
    # Initialize and start method
    method = Method(cfg)
    method.start()

    # Save data
    save_data(cfg, method) 

    # Plot figures
    if cfg.plot:
        plot(cfg, method)
     
    # Print end info
    toc = time.time()
    logger.info('\nExecution time = %.3f s\n' %(toc-tic))
    
    # Close all handlers of logger
    unset_logger()
    

def run_replot(config_dict):
    """Replot figures using existing data"""
    tic = time.time()
    
    # Initialize config
    cfg = Config(config_dict)
    
    # Initialize logger for only printing to console
    set_logger(0, 1)
    
    # Print start info
    logger.info('Electronic Structure Calculator: Replot Figures\n\n')   
    
    # Reload data
    data = reload_data(cfg)
    
    # Replot figures
    replot(cfg, data)
    
    # Print end info
    toc = time.time()
    logger.info('\nExecution time = %.3f s\n' %(toc-tic))
    
    # Close all handlers of logger
    unset_logger()   
    

def parse_args():
    """Parse command line arguments"""
    prog = 'esc'
    description = 'Electronic Structure Calculator'
    version = '%(prog)s ' + __version__
    parser = argparse.ArgumentParser(
        prog=prog, description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input-file', dest='input_file', type=str,
        required=True, help='input file name'
    )
    parser.add_argument(
        '--replot',  action='store_true', default=argparse.SUPPRESS,
        help='replot figures using existing data'
    )
    parser.add_argument('--version', action='version', version=version) 

    return parser.parse_args()



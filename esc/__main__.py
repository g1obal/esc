"""
The entry point of esc

Author: Gokhan Oztarhan
Created date: 27/11/2022
Last modified: 04/12/2022
"""

import argparse
import logging
import sys
import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from .__init__ import __version__
from . import resources
from .config import parse_config_file
from .app import run, run_replot


logging.getLogger(__name__).addHandler(logging.NullHandler())


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
        'input', type=str, nargs='?', default='input.ini', 
        help='configuration file name'
    )
    parser.add_argument(
        '--replot',  action='store_true', default=argparse.SUPPRESS,
        help='replot figures using existing data'
    )
    parser.add_argument('--version', action='version', version=version) 

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # If configuration file does not exist,
    # create default configuration file and exit.
    if not os.path.exists(args.input):
        template = pkg_resources.read_text(resources, 'input-default.ini')
        
        with open('input.ini', 'w') as config_file:
            config_file.write(template)
            
        sys.exit('Input file is not found: %s\n' %args.input \
            + 'Default input file is created.')

    # Parse config file
    config_dict = parse_config_file(args.input)

    # Run esc or replot data
    if 'replot' in args:
        run_replot(config_dict)
    else:
        run(config_dict)



"""
Config Parser X

Author: Gokhan Oztarhan
Created date: 22/12/2021
Last modified: 01/12/2022
"""

import os
import sys
import configparser


# In order to use None as a fallback value
_Unset = object()


class ConfigParserX(configparser.ConfigParser):
    _NONE_STATES = ['', 'none', 'None']
    
    def __init__(self):
        super().__init__()
        self.optionxform = lambda option: option  # for case-sensitive parsing
        self.option_dict = {}
        self._CONVERT = {
            bool: self.getboolean,
            str: self.get,
            int: self.getint,
            float: self.getfloat,
        }    
    
    def parse_file(self, fname):
        if os.path.exists(fname):
            self.read(fname)
        else:
            sys.exit('Config file is not found: %s\n' %fname)
    
    def set_type(
        self, section, option, _type, can_be_None=False, fallback=_Unset
    ):                   
        try:
            if can_be_None and self.get(section, option) in self._NONE_STATES:
                self.option_dict[option] = None
            else:
                self.option_dict[option] = self._CONVERT[_type](section, option)

        except ValueError:
            if fallback is _Unset:
                self._terminate(section, option)  
            else:
                self.option_dict[option] = fallback    
                         
        except (configparser.Error, KeyError):
            self._terminate(section, option)

    def _terminate(self, section, option):
        print('Config file is corrupted.')
        print('section: ', section)
        print('option: ', option)
        sys.exit()
 

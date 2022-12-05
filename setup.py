"""
Electronic Structure Calculator

Author: Gokhan Oztarhan
esc created date: 28/08/2019
Setup module created date: 28/12/2021
Setup module last modified: 05/12/2022
"""

from distutils.core import setup
import re

from esc import __version__


# Parse long_description.
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Parse requirements. Keep only package names. The following are the minimal
# requirements and do not include version numbers (see README.md).
with open('requirements.txt', 'r', encoding='utf-8') as f:
    delimeters = '<|<=|==|!=|~=|>=|>'
    requirements = [
        re.split(delimeters, line)[0].strip() \
        for line in f.readlines() if not line.startswith('#')
    ]

# Initialize setup
setup(
    name = 'esc',
    version = __version__,
    description = 'Electronic Structure Calculator',
    long_description = long_description,
    author = 'Gokhan Oztarhan',
    author_email = 'gooztarhan@gmail.com',
    url = 'https://github.com/g1obal/esc',
    license = 'MIT License',
    license_files = ['LISENCE'],
    keywords = [
        'python3', 'electronic structure', 'quantum mechanics', 
        'tight-binding', 'mean-field Hubbard', 'Hubbard model'
    ],
    packages = [
        'esc', 
        'esc.latgen', 
        'esc.method', 
        'esc.resources',
    ],
    package_dir = {
        'esc': 'esc', 
        'esc.latgen': 'esc/latgen',
        'esc.method': 'esc/method', 
        'esc.resources': 'esc/resources',
    },
    package_data = {'esc.resources': ['input-default.ini']},
    include_package_data = True,
    install_requires = requirements,
)



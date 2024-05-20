"""
Electronic Structure Calculator

Author: Gokhan Oztarhan
esc created date: 28/08/2019
Setup module created date: 28/12/2021
Setup module last modified: 20/05/2024
"""

from distutils.core import setup
import re


# Parse version number.
with open('esc/__init__.py', 'r') as f:
    __version__ = [line.split('=')[-1].strip().replace("'",'') \
        for line in f.readlines() if '__version__' in line][0]

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
    ],
    package_dir = {
        'esc': 'esc', 
        'esc.latgen': 'esc/latgen',
    },
    scripts = ['scripts/esc_run'],
    install_requires = requirements,
)



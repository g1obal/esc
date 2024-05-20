# Electronic Structure Calculator

Basic electronic structure calculations using diagonalization of tight-binding
or mean-field Hubbard Hamiltonians, currently for 2D graphene-like systems.
The program can also produce orbital coefficients for formation of Slater 
determinants to be used in other methods or programs (e.g. for 
[CHAMP](https://github.com/g1obal/CHAMP)).

- **Tight-binding Hamiltonian**
```math
H_{TB} = -t\sum_{\langle i,j \rangle,\sigma} 
c_{i\sigma}^{\dagger} c_{j\sigma} + h.c.
```
where $c_{i\sigma}^{\dagger}$ and $c_{i\sigma}$ are the creation and 
annihiliation operators for $i^{\textrm{th}}$ electron with $\sigma$ spin, and 
$h.c.$ is the Hermitian conjugate of the operators. Additionally, 
$\langle i,j \rangle$ indicates summation over the nearest neighbor lattice
sites, and $t$ is the hopping parameter between the nearest neighbor sites.

- **Mean-field Hubbard Hamiltonian**
```math
H_{MFH} = -t\sum_{\langle i,j \rangle, \sigma} 
c_{i\sigma}^{\dagger} c_{j\sigma}
+ U\sum_{i, \sigma} \langle n_{i\sigma} \rangle n_{i,-\sigma} + h.c.
```
where $n_{i\sigma} = c_{i\sigma}^{\dagger} c_{i\sigma}$ and $U$ is the Coulomb
interaction between the electrons in the same site. 

- **Extended Mean-field Hubbard Hamiltonian**
```math
H_{MFH} = -t\sum_{\langle i,j \rangle, \sigma} 
c_{i\sigma}^{\dagger} c_{j\sigma}
+ U\sum_{i, \sigma} \langle n_{i\sigma} \rangle n_{i,-\sigma}
+ \frac{1}{2} \sum_{\substack{i,j \\ i \neq j}} V_{ij} 
\langle n_{i} \rangle n_{j} + h.c.
```
where $V_{ij}$ is the long range interactions between the electrons in the 1st
nearest neighbor sites and beyond.

The program uses a *half-filling* mean-field Hubbard Hamiltonian which has the
following form.
```math
H_{MFH} = -t\sum_{\langle i,j \rangle, \sigma} 
c_{i\sigma}^{\dagger} c_{j\sigma}
+ U\sum_{i, \sigma} (\langle n_{i\sigma} \rangle - \frac{1}{2}) n_{i,-\sigma}
+ \frac{1}{2} \sum_{\substack{i,j \\ i \neq j}} V_{ij} 
(\langle n_{i} \rangle - 1) n_{j} + h.c.
```
Additionally, 2nd nearest neighbor hopping parameter $t'$ (for tight-binding
part) may be set if required. 

## Requirements
numpy <br />
matplotlib <br />
networkx <br />

This program is expected to work with various versions of the dependencies. 
However, if you notice any inconsistencies, see *requirements.txt* for tested
versions.

## Installation
Installation is not required as long as the environment meets the 
library/package requirements.

Download or clone this repository.
```
$ git clone https://github.com/g1obal/esc.git
```

*Optional:* Install using pip.
```
$ cd esc
$ pip install .
```

## Usage
Usage if installed:
```
$ esc_run
```

Usage without installation:
```
$ python3 esc_run.py
```

The program requires an input file to set configurations (e.g. lattice geometry,
run mode or Hamiltonian parameters). See *examples* directory for input file
examples.

For more information about usage:
```
$ python3 esc_run.py --help
```

## Contact
For questions or bug reports, please create an issue on this repository or 
contact me via [e-mail](mailto:gooztarhan@gmail.com).

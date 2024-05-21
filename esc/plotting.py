"""
Figure plotting module

Author: Gokhan Oztarhan
Created date: 24/07/2019
Last modified: 21/05/2024
"""

import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FixedLocator, LinearLocator, MultipleLocator, FormatStrFormatter
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import networkx


logger = logging.getLogger(__name__)
logging.getLogger('numexpr').setLevel(logging.WARNING) # mute np.numexpr info

plt.ioff()
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}'
})


def plot(cfg, method):
    plot_summary(
        cfg.mode, cfg.eunit, cfg.lunit,
        method.E_up, method.E_dn, cfg.n_up, cfg.n_dn, 
        cfg.a, cfg.a_nau, cfg.pos, cfg.ind_NN, method.V_up, method.V_dn,
        cfg.n_up_start, cfg.n_up_end, cfg.n_dn_start, cfg.n_dn_end,
        cfg.t, cfg.t_nau, cfg.tp, cfg.tp_nau, 
        cfg.U, cfg.U_nau, cfg.U_long_range, cfg.U1_U2_scaling,
        method.E_total, method.E_total_nau,
        cfg.root_dir, plot_E_limit=cfg.plot_E_limit, 
        dos_kde_sigma=cfg.dos_kde_sigma, psi2_kde_sigma=cfg.psi2_kde_sigma,
        mesh_resolution=cfg.mesh_resolution,
        fname=cfg.plot_fname, dpi=cfg.plot_dpi, _format=cfg.plot_format
    )
   
                   
def replot(cfg, data):
    if data['mode'] == 'tb':
        E_up, E_dn = data['E'], data['E']
        V_up, V_dn = data['V'], data['V']
        U = None
        U_nau = None
        U_long_range = None
        U1_U2_scaling = None
        
    elif data['mode'] == 'mfh':
        E_up, E_dn = data['E_up'], data['E_dn']
        V_up, V_dn = data['V_up'], data['V_dn']
        U = data['U']
        U_nau = data['U_nau']
        U_long_range = data['U_long_range']
        U1_U2_scaling = data['U1_U2_scaling']
    
    plot_summary(
        data['mode'], data['eunit'], data['lunit'],
        E_up, E_dn, data['n_up'], data['n_dn'], 
        data['a'], data['a_nau'], data['pos'], data['ind_NN'], V_up, V_dn, 
        cfg.n_up_start, cfg.n_up_end, cfg.n_dn_start, cfg.n_dn_end,
        data['t'], data['t_nau'], data['tp'], data['tp_nau'], 
        U, U_nau, U_long_range, U1_U2_scaling,
        data['E_total'], data['E_total_nau'],
        cfg.root_dir, plot_E_limit=cfg.plot_E_limit, 
        dos_kde_sigma=cfg.dos_kde_sigma, psi2_kde_sigma=cfg.psi2_kde_sigma,
        mesh_resolution=cfg.mesh_resolution, 
        fname=cfg.plot_fname, dpi=cfg.plot_dpi, _format=cfg.plot_format
    )

#------------------------------------------------------------------------------

def kde_1d(val, x, sigma, norm=False):
    """Modified gaussian kernel density estimator for density of states"""
    twovar = 2.0 * sigma**2
    y = np.zeros(x.shape[0])
    
    for i in range(val.shape[0]):
        kernel = np.exp(-(x - val[i])**2 / twovar)
        y += kernel  
        
    if norm:
        y = (y - y.min()) / (y.max() - y.min())
        
    return y


def kde_2d(pos, weight, x, y, sigma):
    """
    Modified gaussian kernel density estimator for electron density
    
    This algorithm using (x, y) instead of (xx, yy) is faster.
    The number of numpy.exp() operations is len(x) * len(y) times less. 
    Instead, there is a numpy.outer() function, which only includes 
    multiplication operations. The cpu times are tested.
    """
    twovar = 2.0 * sigma**2
    z = np.zeros([x.shape[0],y.shape[0]])
    
    for i in range(pos.shape[0]):
        kernel_x = np.exp(-(x - pos[i,0])**2 / twovar)
        kernel_y = np.exp(-(y - pos[i,1])**2 / twovar)
        kernel = np.outer(kernel_y,kernel_x)
        z += kernel * weight[i]

    return z


def lattice_network(pos, ind_NN):  
    """
    Lattice network
    Connect the lattice points with lines
    """
    lattice_net = np.zeros([pos.shape[0], pos.shape[0]])
    lattice_net[ind_NN[:,0],ind_NN[:,1]] = 1
    lattice_net[ind_NN[:,1],ind_NN[:,0]] = 1   
    lattice_net = networkx.Graph(lattice_net)
    for i in range(pos.shape[0]):
        lattice_net.add_node(i, pos=(pos[i,0], pos[i,1]))
        
    node_positions = networkx.get_node_attributes(lattice_net, 'pos')   
    
    return lattice_net, node_positions
    
#------------------------------------------------------------------------------  

def electron_densities(
    a, pos, V_up, V_dn, 
    n_up_start, n_up_end, n_dn_start, n_dn_end,
    kde_sigma, mesh_resolution
):
    tic = time.time()
    
    # KDE sigma
    if kde_sigma == None:
        kde_sigma = 0.2 * a
    else:
        kde_sigma *= a
        
    # Axis limits
    xlim = [pos[:,0].min() - 3 * kde_sigma, pos[:,0].max() + 3 * kde_sigma]
    ylim = [pos[:,1].min() - 3 * kde_sigma, pos[:,1].max() + 3 * kde_sigma]
    meshlim = max([max(xlim), max(ylim)])
    
    # Calculate mesh points
    x = np.linspace(-meshlim, meshlim, mesh_resolution)
    y = np.linspace(-meshlim, meshlim, mesh_resolution)
    xx, yy = np.meshgrid(x, y)

    # Up electrons
    probs = np.conj(V_up) * V_up # there is no transpose since this is psi^2
    weight = probs[:,n_up_start:n_up_end].sum(axis=1)
    zz_up = kde_2d(pos, weight, x, y, kde_sigma)
    
    # Down electrons
    probs = np.conj(V_dn) * V_dn # there is no transpose since this is psi^2
    weight = probs[:,n_dn_start:n_dn_end].sum(axis=1)
    zz_dn = kde_2d(pos, weight, x, y, kde_sigma)
    
    toc = time.time()
    logger.info('electron_densities done. (%.3f s)\n' %(toc - tic)) 

    return xlim, ylim, xx, yy, zz_up, zz_dn

#------------------------------------------------------------------------------

def plot_summary(
    mode, eunit, lunit, E_up, E_dn, n_up, n_dn, 
    a, a_nau, pos, ind_NN, V_up, V_dn,
    n_up_start, n_up_end, n_dn_start, n_dn_end,
    t, t_nau, tp, tp_nau, U, U_nau, U_long_range,
    U1_U2_scaling, E_total, E_total_nau,
    root_dir, plot_E_limit=None, 
    dos_kde_sigma=None, psi2_kde_sigma=None, mesh_resolution=500,
    fname='summary', dpi=600, _format='jpg'
):
    tic_all = time.time()
    
    # Print info start
    logger.info('[plotting]\n----------\n')
    
    # Initialize figure
    fig = plt.figure(figsize=plt.figaspect(0.80))

    # Initialize GridSpec
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.8, 1, 0.5]) 
    
    # Initialize ax list
    ax = []
    ax.append(fig.add_subplot(gs[0,0], box_aspect=1.0))
    ax.append(fig.add_subplot(gs[1,0], box_aspect=1.0))
    ax.append(fig.add_subplot(gs[0,1], adjustable='box', aspect=1.0))
    ax.append(fig.add_subplot(gs[1,1], adjustable='box', aspect=1.0))
    ax.append(fig.add_subplot(gs[0,2], adjustable='box', aspect=1.0))
    ax.append(fig.add_subplot(gs[1,2], adjustable='box', aspect=1.0))
    
    # Set edge tickness of all axes
    for _ax in ax:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(0.40)
        _ax.tick_params(width=0.40, length=2.0)
    
    # Calculate energy divided by t
    E_up_t = E_up / t
    E_dn_t = E_dn / t
    
    # Plot energy spectrum
    ax[0] = plot_E(ax[0], mode, E_up_t, E_dn_t, n_up, n_dn, limit=plot_E_limit)

    # Plot density of states
    ax[1] = plot_dos(ax[1], mode, E_up_t, E_dn_t, kde_sigma=dos_kde_sigma)
    
    # Calculate lattice network (connect the lattice points with lines)
    lattice_net, node_positions = lattice_network(pos, ind_NN)
    
    # Calculate electron densities
    xlim, ylim, xx, yy, zz_up, zz_dn = electron_densities(
        a, pos, V_up, V_dn,
        n_up_start, n_up_end, n_dn_start, n_dn_end, 
        psi2_kde_sigma, mesh_resolution
    )
    
    # Plot densities
    fig, ax[2] = plot_density(
        fig, ax[2], xlim, ylim, lattice_net, node_positions,
        xx, yy, zz_up + zz_dn, 'Total Electron Density, $|\\psi|^{2}$'
    )
    fig, ax[3] = plot_density(
        fig, ax[3], xlim, ylim, lattice_net, node_positions,
        xx, yy, zz_up - zz_dn, 'Spin Density'
    )
    fig, ax[4] = plot_density_updn(
        fig, ax[4], xlim, ylim, lattice_net, node_positions,
        xx, yy, zz_up, 'Up Electron Density, $|\\psi_{\\uparrow}|^{2}$'
    )
    fig, ax[5] = plot_density_updn(
        fig, ax[5], xlim, ylim, lattice_net, node_positions,
        xx, yy, zz_dn, 'Down Electron Density, $|\\psi_{\\downarrow}|^{2}$'
    )
    
    # Info ax
    ax_info = fig.add_subplot(1,1,1)
    ax_info.clear()
    ax_info.axis('off')
    info_str = 'mode = %s, ' %mode \
        + 'n\_site = %i, ' %pos.shape[0] \
        + 'n\_elec = %i' %(n_up + n_dn) \
        + '\nn\_up = %i, ' %n_up \
        + 'n\_dn = %i, ' %n_dn \
        + 'Sz = %.2f' %(0.5 * (n_up - n_dn)) \
        + '\n$a$ = %.5e (%.5f %s)' %(a, a_nau, lunit)
    info_str = info_str.replace('e-', '$e-$').replace('e+', '$e+$')
    ax_info.text(0.51, 1.115, info_str, ha='center', va='top', fontsize=5.25)
    
    info_str = '$t$ = %.5e (%.5f %s)\n' %(t, t_nau, eunit) \
        + '$tp$ = %.5e (%.5f %s)' %(tp, tp_nau, eunit)
    if mode == 'mfh':
        info_str += '\n$U$ = %.5e (%.5f %s)\n' %(U, U_nau, eunit) \
            + '$U / t$ = %.5f' %(U / t)
    info_str = info_str.replace('e-', '$e-$').replace('e+', '$e+$')
    ax_info.text(0.0, 1.115, info_str, ha='left', va='top', fontsize=5.25)
    
    info_str = 'E\_total = %.15f\n' %E_total \
        + '(%.15f %s)' %(E_total_nau, eunit)
    if mode == 'mfh':
        info_str += '\n\nU\_long\_range = %s\n' %U_long_range \
            + 'U1\_U2\_scaling = %s' %U1_U2_scaling
    ax_info.text(1.0, 1.115, info_str, ha='right', va='top', fontsize=5.25)
    
    # Add padding between subplots
    fig.tight_layout()
    #plt.subplots_adjust(wspace=0.4, hspace=0.25)
    
    # Save figure
    fname = os.path.join(root_dir, fname) + '.' + _format
    fig.savefig(fname, dpi=dpi, format=_format, bbox_inches='tight') 
    plt.close(fig)
    
    toc_all = time.time()
    logger.info('plot_summary done. (%.3f s)\n' %(toc_all - tic_all)) 


#------------------------------------------------------------------------------

def plot_E(ax, mode, E_up, E_dn, n_up, n_dn, limit=None):
    tic = time.time()
    
    E_shape = E_up.shape[0]
    
    if limit is None:
        if E_shape <= 78:
            limit = (-0.44667521 * np.log(E_shape) + 2.38053782) * E_up.max()
        elif E_shape <= 762:
            limit = (-0.11001269 * np.log(E_shape) + 0.90267936) * E_up.max()
        elif E_shape <= 3282:
            limit = (-0.06401283 * np.log(E_shape) + 0.60973256) * E_up.max()
        elif E_shape <= 10806:
            limit = (-0.02693211 * np.log(E_shape) + 0.31730567) * E_up.max()
        else:
            limit = 0.05 * E_up.max()

    # Axis limits
    xlim_up = [np.where(E_up > -limit)[0][0], np.where(E_up < limit)[0][-1]]
    ylim_up = [E_up[E_up > -limit].min(), E_up[E_up < limit].max()]
    
    xlim_dn = [np.where(E_dn > -limit)[0][0], np.where(E_dn < limit)[0][-1]]
    ylim_dn = [E_dn[E_dn > -limit].min(), E_dn[E_dn < limit].max()]
    
    xlim = [min(xlim_up[0], xlim_dn[0]), max(xlim_up[1], xlim_dn[1])]
    ylim = [min(ylim_up[0], ylim_dn[0]), max(ylim_up[1], ylim_dn[1])]  
    
    # Axis adjustments
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%i'))
    #ax.xaxis.set_major_locator(LinearLocator(6))
    ax.tick_params(labelsize=5.5)
    ax.set_xlabel('Energy index (zero-based)', fontsize=6.5)
    ax.set_ylabel('Energy / $t$', fontsize=6.5)
    
    # PLot figure
    x = range(E_up.shape[0])
    ax.plot(x, np.zeros(E_up.shape[0]), '--k', lw=0.55, alpha=0.5)
    ax.plot(x, E_up, 'bo', markersize=2.25, label='$\\epsilon_{\\uparrow}$')
    if mode == 'mfh':
        ax.plot(
            x, E_dn, 'ro', markersize=2.60, markeredgewidth=0.6,
            markerfacecolor='none', label='$\\epsilon_{\\downarrow}$'
        )
        legend = ax.legend(loc='best', fontsize=6)
        legend.get_frame().set_linewidth(0.5)
        
    ind_up = n_up - 1
    ind_dn = n_dn - 1
    ax.text(ind_up, E_up[ind_up], str(ind_up), c='g', fontsize=5.25)
    ax.text(ind_dn, E_dn[ind_dn], str(ind_dn), c='g', fontsize=5.25)
    
    toc = time.time()
    logger.info('plot_E done. (%.3f s)\n' %(toc - tic)) 

    return ax
 
 
def plot_dos(ax, mode, E_up, E_dn, kde_sigma=None):
    tic = time.time()
    
    if kde_sigma is None:
        kde_sigma_up = np.diff(E_up).max() / 10
        kde_sigma_dn = np.diff(E_dn).max() / 10
        kde_sigma = max([kde_sigma_up, kde_sigma_dn])

    # Axis limits
    xlim_up = [E_up.min() * 1.1, E_up.max() * 1.1]
    xlim_dn = [E_dn.min() * 1.1, E_dn.max() * 1.1]
    xlim = [min(xlim_up[0], xlim_dn[0]), max(xlim_up[1], xlim_dn[1])]
    
    # DOS
    x = np.linspace(xlim[0], xlim[1], E_up.shape[0] * 10)
    y_up = kde_1d(E_up, x, kde_sigma, norm=True)  
    if mode == 'mfh':
        y_dn = kde_1d(E_dn, x, kde_sigma, norm=True)  

    # Axis adjustments
    ax.set_xlim(*xlim)
    ax.set_ylim([0,1])
    #ax.yaxis.set_major_locator(LinearLocator(5))
    #ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    ax.tick_params(labelsize=5.5)
    ax.set_xlabel('Energy / $t$', fontsize=6.5)
    ax.set_ylabel('Density of States', fontsize=6.5)
    
    # Plot figure
    ax.plot(x, y_up, 'b-', lw=0.8, label='$\\sigma_{\\uparrow}$')
    if mode == 'mfh':
        ax.plot(x, y_dn, 'r-.', lw=0.8, label='$\\sigma_{\\downarrow}$')
        legend = ax.legend(loc='best', fontsize=6)
        legend.get_frame().set_linewidth(0.5)
        
    toc = time.time()
    logger.info('plot_dos done. (%.3f s)\n' %(toc - tic)) 
    
    return ax
    
    
def plot_density(
    fig, ax, xlim, ylim, lattice_net, node_positions, xx, yy, zz, title
):
    tic = time.time()
    
    # Colormap (seismic, bwr, RdYlBu, afmhot, jet) and line settings
    if title == 'Spin Density':
        cmap = 'seismic'  
        zmax = abs(zz).max().max()
        zlim = [-zmax, zmax] # z-axis special limit for spin density
    else:
        cmap = 'jet' 
        zlim = [zz.min().min(), zz.max().max()]
    latticecolor = '#616161' # (#616161:gray)
    latticealpha = 0.5
    latticewidth = 0.5
    shading = 'nearest' # shading = 'auto', 'nearest', 'gouraud'
    
    # Axis adjustments
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.tick_params(labelsize=6)
    #ax.set_xlabel('x', fontsize=6)
    #ax.set_ylabel('y', fontsize=6)
    ax.set_title(title, fontsize=6) 
    
    # Plot surface and lattice network
    surf = ax.pcolormesh(
        xx, yy, zz, vmin=zlim[0], vmax=zlim[1], cmap=cmap, shading=shading
    )
    networkx.draw_networkx(
        lattice_net, pos=node_positions, node_size=0, with_labels=False, 
        width=latticewidth, edge_color=latticecolor, alpha=latticealpha, ax=ax
    )
    
    # Add colorbar
    # create a colorbar ax on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(surf, cax=cax, ticks=LinearLocator(5))
    cbar.ax.tick_params(width=0.40, length=2.00, labelsize=4)
    cbar.outline.set_linewidth(0.40)
    cbar.ax.yaxis.get_offset_text().set(size=5) # for offset text (e.g. 10^-5)
    #cbar_ticks = np.linspace(zlim[0], zlim[1], 5)
    #cbar.set_ticks(cbar_ticks)
    #cbar.formatter.set_powerlimits((-3, 4)) 
    
    logger_info_title = title.split(',')[0]
    toc = time.time()
    logger.info('plot_density "%s" done. (%.3f s)\n' \
        %(logger_info_title, toc - tic)) 
    
    return fig, ax
    
    
def plot_density_updn(
    fig, ax, xlim, ylim, lattice_net, node_positions, xx, yy, zz, title
):
    tic = time.time()
    
    # Colormap (seismic, bwr, RdYlBu, afmhot, jet) and line settings
    cmap = 'jet' 
    zlim = [zz.min().min(), zz.max().max()]
    latticecolor = '#616161' # (#616161:gray)
    latticealpha = 0.5
    latticewidth = 0.25
    shading = 'nearest' # shading = 'auto', 'nearest', 'gouraud'
    
    # Axis adjustments
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.tick_params(labelsize=3)
    #ax.set_xlabel('x', fontsize=3)
    #ax.set_ylabel('y', fontsize=3)
    ax.set_title(title, fontsize=3.25) 
    
    # Plot surface and lattice network
    surf = ax.pcolormesh(
        xx, yy, zz, vmin=zlim[0], vmax=zlim[1], cmap=cmap, shading=shading
    )
    networkx.draw_networkx(
        lattice_net, pos=node_positions, node_size=0, with_labels=False, 
        width=latticewidth, edge_color=latticecolor, alpha=latticealpha, ax=ax
    )
    
    # Add colorbar
    # create a colorbar ax on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(surf, cax=cax, ticks=LinearLocator(5))
    cbar.ax.tick_params(width=0.25, length=1.125, labelsize=2)
    cbar.outline.set_linewidth(0.25)
    cbar.ax.yaxis.get_offset_text().set(size=2.25)
    #cbar_ticks = np.linspace(zlim[0], zlim[1], 5)
    #cbar.set_ticks(cbar_ticks)
    #cbar.formatter.set_powerlimits((-3, 4)) 
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.25)
    ax.tick_params(width=0.25, length=1.125)
    
    logger_info_title = title.split(',')[0]
    toc = time.time()
    logger.info('plot_density_updn "%s" done. (%.3f s)\n' \
        %(logger_info_title, toc - tic)) 
    
    return fig, ax





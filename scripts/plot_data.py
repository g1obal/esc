"""
Plot data

Author: Gokhan Oztarhan
Created date: 02/04/2024
Last modified: 02/04/2024
"""

import os
import sys
from copy import deepcopy
from itertools import cycle, product
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FixedLocator, MaxNLocator


DATA_FILE_NAME = 'data.csv'
SORT_COLS = ['mode', 'U/t', 'n_side']

GROUPBY_COLS = ['mode', 'U/t']

PLOT_DIR = 'data_plots'

DPI = 200
PLOT_FORMAT = 'png'
FONTSIZE_XYLABEL = 16
LABELSIZE_TICK_PARAMS = 12
FONTSIZE_LEGEND = 10
DRAW_LEGEND = True
DRAW_TITLE = False

# Set LEGEND_ORDER to None for plotting all groups regardless of order,
# as well as the groups that are not in this list
LEGEND_ORDER = None
LEGEND = {
    'tb': 'Tight-binding',
    'mfh': 'Mean-field Hubbard',
}

warnings.filterwarnings('ignore') # Suppress warnings!

plt.ioff()
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}'
})

if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)
   

def plot_data():
    # Load data and sort
    df = pd.read_csv(DATA_FILE_NAME, header=0, index_col=0)
    df = df.sort_values(SORT_COLS, ignore_index=True)

    # Group data      
    df_grouped = df.groupby(GROUPBY_COLS)
  
    plot(
        df_grouped, 'n_side', 'E_total',
        xlabel='n_side', ylabel='Total Energy (au)',
        title='Total Energy vs n_side',
        groups=LEGEND_ORDER,
    )
    
    plot(
        df_grouped, 'n_side', 'p_edge_pol',
        xlabel='n_side', ylabel='$p$ (Edge Pol.)',
        title='$p$ (Edge Pol.) vs n_side',
        groups=LEGEND_ORDER,
    )


def plot(
    df_grouped, 
    xfeature, yfeature, errfeature=None, 
    xlabel=None, ylabel=None, 
    xtype=float, ytype=float,
    title=None,
    groups=None,
    connect_group_points=True,
    markersize=3,
    elinewidth=1, 
    capsize=2,
    df_background_line=None,
    fname=None
):     
    if xlabel is None:
        xlabel = xfeature.replace('_', '\_')
    if ylabel is None:
        ylabel = yfeature.replace('_', '\_')
        
    if groups is not None:
        groups = [
            groupname for groupname in groups \
            if groupname in df_grouped.groups
        ]
    else:
        groups = df_grouped.groups

    # Initialize line style cycler
    line_cycler = _linestyle()
    
    # Initialize figure
    fig = plt.figure(figsize=plt.figaspect(1.0))
    
    # Initialize ax list
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    
    # Draw background line
    if df_background_line is not None:
        x = get_feature(df_background_line, xfeature, xtype)
        y = get_feature(df_background_line, yfeature, ytype)
        ax[-1].plot(x, y, '--k', alpha=0.5)
    
    for groupname in groups:     
        df = df_grouped.get_group(groupname)
    
        style = next(line_cycler)
        linestyle = style['linestyle']
        color = style['color']
        legend = LEGEND.get(groupname, groupname)
        if isinstance(legend, tuple):
            legend = ', '.join(str(lgn) for lgn in legend).replace('_', '\_')
        
        x = get_feature(df, xfeature, xtype)
        y = get_feature(df, yfeature, ytype)
        
        if errfeature is not None:
            yerr = df[errfeature]
            # If all elements of yerr are NaN, an error is raised.
            if np.isnan(yerr).all():
                yerr = None
        else:
            yerr = None
        
        if not connect_group_points:
            linestyle[1] = 'none'
        
        ax[-1].errorbar(
            x, y, yerr, label=legend,
            color=color, linestyle=linestyle[1], 
            fmt=linestyle[0], markersize=markersize,
            elinewidth=elinewidth, capsize=capsize
        )
    
    # Get all x values if group points are not connected
    if df_background_line is not None and not connect_group_points:
        x = get_feature(df_background_line, xfeature, xtype)

    # x-axis locators
    if xtype == str:
        ax[-1].tick_params(axis='x', rotation=90)
    else:
        ax[-1].xaxis.set_major_locator(FixedLocator(x)) 
        #ax[-1].xaxis.set_major_locator(FixedLocator([10, 15, 20, 25, 30, 35])) 
        #ax[-1].xaxis.set_major_locator(LinearLocator(5)) 
        #ax[-1].xaxis.set_major_locator(MaxNLocator(nbins=20)) 
        #ax[-1].tick_params(axis='x', rotation=45)
        #ax[-1].ticklabel_format(axis='x', style='scientific',scilimits=(0,0))
        #ax[-1].tick_params(axis='x', labelsize=LABELSIZE_TICK_PARAMS)
    
    # y-axis locators
    #ax[-1].yaxis.set_major_locator(LinearLocator(5))
    #ax[-1].yaxis.set_major_locator(MaxNLocator(nbins=6)) 
    #ax[-1].tick_params(axis='y', labelsize=LABELSIZE_TICK_PARAMS)
    
    ax[-1].tick_params(labelsize=LABELSIZE_TICK_PARAMS)
    
    ax[-1].set_xlabel(xlabel, fontsize=FONTSIZE_XYLABEL)
    ax[-1].set_ylabel(ylabel, fontsize=FONTSIZE_XYLABEL)
        
    if DRAW_LEGEND:
        ax[-1].legend(
            loc='best',
            bbox_to_anchor =(1.0, 1.0), 
            ncol = 1, 
            fontsize=FONTSIZE_LEGEND
        )
        
    if DRAW_TITLE and title is not None:
        ax[-1].set_title(title)
    
    if fname is None:
        filename = os.path.join(PLOT_DIR, yfeature + '.' + PLOT_FORMAT)
    else:
        filename = os.path.join(PLOT_DIR, fname + '.' + PLOT_FORMAT)
        
    fig.savefig(filename, dpi=DPI, format=PLOT_FORMAT, bbox_inches='tight') 
    plt.close(fig)
    
    print('Done plot: %s' %filename)
    
    
def get_feature(df, feature, _type):
    if _type == str:
        x = df[feature].apply(lambda string: string.replace('_', '\_'))
    else:
        x = df[feature]
    return x
            

def _linestyle(): 
    linestyle = { 
        'linestyle': [['o', '-'], ['^', '--'], ['s', '-.'], ['h', ':']], 
        'color': ['k', 'r', 'y', 'g', 'c', 'b', 'm'],
    }    
    linestyle = _grid(linestyle)
    return cycle(linestyle)
    

def _grid(params):
    return [dict(zip(params, i)) for i in product(*params.values())]

          
if __name__ == '__main__':
    plot_data()
        


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

font_path = '/usr/share/fonts/truetype/lato/Lato-Semibold.ttf'
prop = font_manager.FontProperties(fname=font_path)
prop.set_size(18)

mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['mathtext.default'] = 'regular'

def truncate_colormap(cmap, minval=0, maxval=1, n=100):
    new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                  cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def format_axis(ax, xlabel, ylabel, props, title=None, xlims=None, ylims=None, nbins=6):
    if title:
        ax.set_title(title, fontproperties=props)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)

    ax.set_xlabel(xlabel, fontproperties=props)
    ax.set_ylabel(ylabel, fontproperties=props)
    ax.tick_params(direction='in', length=6, width=1)
    ax.tick_params(which='minor', direction='in', length=3)
    ax.locator_params(axis='x', nbins=nbins)
    
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(props)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(props)

palette = ['#316986', '#408885', '#AAA850', '#F2B959']
cmap = LinearSegmentedColormap.from_list('cmap', palette)
cmap_temp = LinearSegmentedColormap.from_list('cmap_temp', ['#316986', '#C4C2C3', '#C86646'])
cnorm = mpl.colors.Normalize(vmin=0, vmax=1)
disc_colors = ['#316986', '#F2B959']
disc_colors_light = ['#6A96A9', '#F6CD84']
cmap_disc = ListedColormap(disc_colors)
cmap_disc_light = ListedColormap(disc_colors_light)
set_colors = {'train': '#316986', 'valid': '#C86646', 'test': '#F2B959'}
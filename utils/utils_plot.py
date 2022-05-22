import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# standard formatting for plots
fontsize = 18
textsize = 14
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize


def truncate_colormap(cmap, minval=0, maxval=1, n=100):
	new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
												  cmap(np.linspace(minval, maxval, n)))
	return new_cmap


def format_axis(ax, xlabel, ylabel, title=None, xlims=None, ylims=None, nbins=None):
	if title: ax.set_title(title, fontsize=fontsize)
	if xlims: ax.set_xlim(xlims)
	if ylims: ax.set_ylim(ylims)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.tick_params(direction='in', length=6, width=1)
	ax.tick_params(which='minor', length=0)
	
	if nbins: ax.locator_params(axis='x', nbins=nbins)


def hex_to_rgb(color):
	color = color.strip("#")
	n = len(color)
	return tuple(int(color[i:i+n//3],16) for i in range(0,n,n//3))


def rgb_to_dec(color):
	return [c/256. for c in color]


def make_continuous_cmap(hex_colors, float_values):
	n = len(hex_colors)
	rgb_colors = [rgb_to_dec(hex_to_rgb(color)) for color in hex_colors]
	cdict = {}
	primary = ['red','green','blue']
	for i, p in enumerate(primary):
		cdict[p] = [[float_values[j], rgb_colors[j][i], rgb_colors[j][i]] for j in range(n)]
		cmap = mpl.colors.LinearSegmentedColormap('my_cmap', segmentdata=cdict, N=500)
	return cmap


def plot_colormaps(cmaps):
	gradient = np.linspace(0, 1, 256)
	gradient = np.vstack((gradient, gradient))

	for cmap in cmaps:
		fig, ax = plt.subplots(figsize=(8,0.5))
		ax.imshow(gradient, aspect='auto', cmap=cmap)
		ax.set_yticks([]); ax.set_xticks([]);

# parameters colormap
palette = ['#283C73', '#316986', '#408885', '#AAA850', '#F2B959']
cmap = LinearSegmentedColormap.from_list('cmap', palette)

# temperature colormap
cmap_temp = LinearSegmentedColormap.from_list('cmap_temp', ['#316986', '#C4C2C3', '#C86646'])

# errors colormap
cmap_mse = LinearSegmentedColormap.from_list('cmap_mse', ['#283C73', '#316986', '#C86646', '#C4C2C3'])

# divided colormap
cmap_cm = make_continuous_cmap(['#FFFFFF', '#C4C2C3', '#F2B959', '#316986', '#283C73'], [0,0.499,0.5,0.75,1])

# colors
disc_colors = ['#316986', '#F2B959']
disc_colors_light = ['#6A96A9', '#F6CD84']
cmap_disc = ListedColormap(disc_colors)
cmap_disc_light = ListedColormap(disc_colors_light)
set_colors = {'train': '#316986', 'valid': '#C86646', 'test': '#F2B959'}
spin_colors = {'+': '#316986', '-': '#C86646'}
FM_colors = ['#C5C0B2', '#F8C278', '#FFB95B', '#467A93', '#F7EFDD']
AFM_colors = ['#C5C0B2', '#BDBB75', '#AAA850', '#FFB95B', '#F8C278', '#F7EFDD', '#FFF7E8']

# norm
cnorm = mpl.colors.Normalize(vmin=0, vmax=1)
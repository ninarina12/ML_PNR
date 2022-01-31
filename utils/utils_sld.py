import numpy as np

from scipy.interpolate import interp1d
from utils.utils_plot import *
from utils.utils_simulate import build_inst, build_stack, SLD_sim

def init_BiSe10_EuS5(b, y_pred, y_header, prox=False):
    if prox:
        ys = np.array([y_header.index(k) for k in y_header if k.startswith('s_')])
        y_pred[ys] = 0.1

        ys = [y_header.index(k) for k in y_header if (k.startswith('magn')) and (k != 'magn_prox')]
        y_pred[ys] = 0.

    dens_sub = np.array([[y_pred[y_header.index('dens_sub')]*1e-3]])
    v_sub = np.array([[0., 3., 0.]])

    dens_main1 = np.array([[y_pred[y_header.index('dens_TI')]*1e-3]])
    v_main1 = np.array([[y_pred[y_header.index('d_TI')]*10., 1., 0.],
                        [y_pred[y_header.index('d_prox')]*10., y_pred[y_header.index('s_prox')]*10., y_pred[y_header.index('magn_prox')]]])

    dens_main2 = np.array([[y_pred[y_header.index('dens_FM')]*1e-3]])
    v_main2 = np.array([[y_pred[y_header.index('d_FM')]*10., y_pred[y_header.index('s_FM')]*10., y_pred[y_header.index('magn_FM')]]])

    dens_cap = np.array([[y_pred[y_header.index('dens_cap')]*1e-3]])
    v_cap = np.array([[y_pred[y_header.index('d_cap')]*10., y_pred[y_header.index('s_cap')]*10., 0.]])
    
    # replicate properties for subdivided layers
    dens_main1 = np.repeat(dens_main1, v_main1.shape[0], axis=0)
    dens_main2 = np.repeat(dens_main2, v_main2.shape[0], axis=0)

    # assemble all parameters
    sub = np.concatenate([dens_sub, v_sub], axis=1)
    main1 = np.concatenate([dens_main1, v_main1], axis=1)
    main2 = np.concatenate([dens_main2, v_main2], axis=1)
    cap = np.concatenate([dens_cap, v_cap], axis=1)

    layer_names = [r'$\alpha-Al_2O_3$', '$Bi_2Se_3$', 'EuS', '$a-Al_2O_3$']
    layer_bounds = [0, sum(main1[:,1]), sum(main1[:,1]) + sum(main2[:,1]), sum(main1[:,1]) + sum(main2[:,1]) + sum(cap[:,1])]

    return build_stack(b, sub, main1, main2, cap), layer_names, layer_bounds


def init_CrO20_BiSbTe20(b, y_pred, y_header, prox=False):
    if prox:
        ys = np.array([y_header.index(k) for k in y_header if k.startswith('s_')])
        y_pred[ys] = 0.1

        ys = [y_header.index(k) for k in y_header if (k.startswith('magn')) and (k != 'magn_prox')]
        y_pred[ys] = 0.

    dens_sub = np.array([[y_pred[y_header.index('dens_sub')]*1e-3]])
    v_sub = np.array([[0., 3., 0.]])

    dens_main1 = np.array([[y_pred[y_header.index('dens_AFM')]*1e-3]])
    v_main1 = np.array([[y_pred[y_header.index('d_AFM')]*10., 1., 0.],
                        [y_pred[y_header.index('d_iAFM')]*10., y_pred[y_header.index('s_iAFM')]*10., y_pred[y_header.index('magn_iAFM')]]])

    dens_main2 = np.array([[y_pred[y_header.index('dens_TI')]*1e-3]])
    v_main2 = np.array([[y_pred[y_header.index('d_prox')]*10., 1., y_pred[y_header.index('magn_prox')]],
                        [y_pred[y_header.index('d_TI')]*10., y_pred[y_header.index('s_TI')]*10., 0.]])

    dens_cap = np.array([[y_pred[y_header.index('dens_cap')]*1e-3], [y_pred[y_header.index('dens_ox')]*1e-3]])
    v_cap = np.array([[y_pred[y_header.index('d_cap')]*10., y_pred[y_header.index('s_cap')]*10., 0.],
                      [y_pred[y_header.index('d_ox')]*10., y_pred[y_header.index('s_ox')]*10., 0.]])
    
    # replicate properties for subdivided layers
    dens_main1 = np.repeat(dens_main1, v_main1.shape[0], axis=0)
    dens_main2 = np.repeat(dens_main2, v_main2.shape[0], axis=0)

    # assemble all parameters
    sub = np.concatenate([dens_sub, v_sub], axis=1)
    main1 = np.concatenate([dens_main1, v_main1], axis=1)
    main2 = np.concatenate([dens_main2, v_main2], axis=1)
    cap = np.concatenate([dens_cap, v_cap], axis=1)

    layer_names = [r'$\alpha-Al_2O_3$', '$Cr_2O_3$', '$(Bi,Sb)_2Te_3$', 'Te/$TeO_2$']
    layer_bounds = [0, sum(main1[:,1]), sum(main1[:,1]) + sum(main2[:,1]), sum(main1[:,1]) + sum(main2[:,1]) + sum(cap[:,1])]

    return build_stack(b, sub, main1, main2, cap), layer_names, layer_bounds


def plot_SLD(image_dir, sample_name, q, y_pred, y_header, title, scale=0, y_lims=None, show_prox=False, ncol=1):
    # get sample properties
    sample_sim = __import__('sim_' + sample_name.replace('_', ''))
    inst = build_inst(sample_sim.wavelen)
    b = sample_sim.init_sample()[2][0]
    stack, layer_names, layer_bounds = eval('init_' + sample_name)(b, y_pred, y_header)

    # simulate SLD
    sld = SLD_sim(q/10., inst, stack)

    # resample to N points
    N = len(q)
    z = np.linspace(sld['z'].min(), sld['z'].max(), N)
    re = interp1d(sld['z'], sld['Re non-mag'])(z)
    im = interp1d(sld['z'], sld['Im non-mag'])(z)
    mag = interp1d(sld['z'], sld['mag'])(z)
    z *= 0.1
    layer_bounds = 0.1*np.array(layer_bounds)

    # set up figure and axes
    fig, ax = plt.subplots(figsize=(5,4.5))

    # plot SLD
    if scale:
        mag *= scale
        im *= scale
        tag = ' x ' + str(scale)
    else: tag = ''

    alpha = 0.5
    ax.plot(z, re, color=palette[1], zorder=1)
    ax.plot(z, mag, color=palette[3], zorder=2)
    ax.plot(z, im, color=palette[-1], zorder=4)
    ax.fill_between(z, re, color=palette[1], alpha=alpha - 0.2, label='NSLD', zorder=1)
    ax.fill_between(z, mag, color=palette[3], alpha=alpha, label='MSLD' + tag, zorder=2)
    ax.fill_between(z, im, color=palette[-1], alpha=alpha, label='ASLD' + tag, zorder=4)
    for i in range(len(layer_bounds)):
        ax.axvline(layer_bounds[i], linestyle='--', color='#ADABA4')

    name_pos = (layer_bounds[:-1] + layer_bounds[1:])/2.
    for i, name in enumerate(layer_names[1:]):
        ax.text(name_pos[i], 0.7, name, color='#262626', ha='center', va='center', fontsize=textsize)

    if show_prox:
        # overlay MSLD with 1A roughness
        stack, _, _ = eval('init_' + sample_name)(b, y_pred, y_header, prox=True)
        
        # simulate SLD
        sld_ = SLD_sim(q/10., inst, stack)

        # resample to N points
        z_ = np.linspace(sld_['z'].min(), sld_['z'].max(), N)
        mag = interp1d(sld_['z'], sld_['mag'])(z_)
        z_ *= 0.1
        if scale: mag *= scale
        ax.plot(z_, mag, color='darkgreen', zorder=3)
        ax.fill_between(z_, mag, color='darkgreen', alpha=alpha, label='$MSLD_{prox}$' + tag, hatch='//', zorder=3)

    if not y_lims: y_lims = [im.min() - 0.5, re.max() + 0.5]
    format_axis(ax, 'z (nm)', 'SLD (10$^{-4}$/nm$^2$)', xlims=[z.min(), z.max()], ylims=y_lims)

    ax.legend(edgecolor='white', framealpha=1, loc='upper right', ncol=ncol)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(image_dir + '/sld_' + title + '.pdf', bbox_inches='tight')
    return y_lims
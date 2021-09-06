import numpy as np
import pandas as pd
import os
import json
import seaborn as sns

from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix

from plot_imports import *
from pnr_models import *

#################################################################################################################

def parse_metadata(dir_name):
    # extract relevant parameters from metadata file
    with open(dir_name + '/metadata.txt') as f:
        meta = json.load(f)
        layers = meta['PARAMETERS']['layers']
        rho = meta['PARAMETERS']['densities (g/cm^3)']
        M = meta['PARAMETERS']['molar masses (g/mol)']
        N = meta['PARAMETERS']['number of q-points']
        q_min = meta['PARAMETERS']['q_min (A^-1)']
        q_max = meta['PARAMETERS']['q_max (A^-1)']
    return layers, rho, M, N, q_min, q_max


def parse_labels(df, include_roughness=False):
    # parse and format labels for plotting selected parameters
    
    # drop columns with duplicate values (e.g. dens columns if multiple sublayers)
    cols = [k for k in df.columns if '.' in k]
    df = df.drop(columns=cols)

    # parse y-values
    df['class_prox'] = (df['magn_prox'] > 0).astype(float)
    y_data = df.values
    y_columns = list(df.columns)
    
    # exclude columns whose values are always 0
    y_header = list(df.columns[~(df == 0).all()])
    
    if not include_roughness:
        # exclude roughness
        y_header = [l for l in y_header if not l.startswith('s_')]
        sym_old = ['dens_', 'd_', 'magn_']
        sym_new = [r'\rho_', 't_', 'm_']
    else:
        sym_old = ['dens_', 'd_', 's_', 'magn_']
        sym_new = [r'\rho_', 't_', r'\sigma_', 'm_']
    
    y_ids = np.array([list(df.columns).index(l) for l in y_header])

    # translate to symbols
    y_labels = [r'$' + l.split('_')[0] + '_{' + l.split('_')[1] + '}$' for l in y_header]
    for old, new in zip(sym_old, sym_new):
        y_labels[:-1] = [l.replace(old, new) for l in y_labels[:-1]]

    y_units = ['(f.u./'+r'$nm^3$)']*len([l for l in y_header if l.startswith('dens_')]) + \
              ['(nm)']*len([l for l in y_header if l.startswith('d_')]) + \
              ['(nm)']*len([l for l in y_header if l.startswith('s_')]) + \
              [r'($\mu_B$'+'/f.u.)']*len([l for l in y_header if l.startswith('magn_')]) + ['']

    return y_data, y_columns, y_header, y_ids, y_labels, y_units


def read_exp(file_name):
    # read experimental data
    df = pd.read_csv(file_name, usecols=[0,1,2,3], names=['Q','R','dR','dQ'], skiprows=24, sep='\t', dtype=float)
    return df

#################################################################################################################

def process_data(data_dir, sample_name, exp_name, q, seed, include_roughness=False):
    # read and process simulation data
    xdata_uu =  data_dir + '/xdata_uu.txt'
    xdata_dd =  data_dir + '/xdata_dd.txt'
    ydata =  data_dir + '/ydata.txt'

    x_uu = pd.read_csv(xdata_uu, header=None, sep=' ', dtype=float).values
    x_dd = pd.read_csv(xdata_dd, header=None, sep=' ', dtype=float).values
    x_orig = np.stack([x_uu, x_dd], axis=2)
    
    # add noise based on experimental spectra
    xdata_uu =  data_dir + '/xdata_uu_exp.txt'
    xdata_dd =  data_dir + '/xdata_dd_exp.txt'
    x_uu = pd.read_csv(xdata_uu, header=None, sep=' ', dtype=float).values
    x_dd = pd.read_csv(xdata_dd, header=None, sep=' ', dtype=float).values
    x_data = np.stack([x_uu, x_dd], axis=2)
    x_data = add_noise(exp_name, q, x_data, x_orig, seed)

    # normalize
    x_data, x_moms = normalize_log(x_data)
    x_orig, _ = normalize_log(x_orig, x_moms)

    # read labels
    y_df = pd.read_csv(ydata, dtype=float)
    y_data, y_columns, y_header, y_ids, y_labels, y_units = parse_labels(y_df, include_roughness=include_roughness)

    return x_data, x_orig, x_moms, y_data, y_columns, y_header, y_ids, y_labels, y_units


def process_exp(exp_names, q, x_moms=(0,1)):
    # read and normalize experimental data
    x_exp_list = []

    for i in range(len(exp_names)):
        xdata_uu = 'experiments/' + exp_names[i] + '/x_uu.dat'
        xdata_dd = 'experiments/' + exp_names[i] + '/x_dd.dat'
        df_uu = read_exp(xdata_uu)
        df_dd = read_exp(xdata_dd)

        # resample to q points
        x_uu = interp1d(df_uu['Q'].values, df_uu['R'].values, fill_value='extrapolate')(q/10.)
        x_dd = interp1d(df_dd['Q'].values, df_dd['R'].values, fill_value='extrapolate')(q/10.)

        # apply gentle smoothing
        window_radius = int(len(q)/32.)
        x_uu = smooth_inverse(q, x_uu, window_radius)
        x_dd = smooth_inverse(q, x_dd, window_radius)

        x_uu = np.expand_dims(x_uu, axis=0)
        x_dd = np.expand_dims(x_dd, axis=0)
        x_exp = np.stack([x_uu, x_dd], axis=2)

        # normalize
        x_exp, _ = normalize_log(x_exp, x_moms)
        
        x_exp_list += [x_exp]

    x_exp = np.concatenate(x_exp_list, axis=0)

    return x_exp


def add_noise(exp_name, q, x, x_orig, seed):
    np.random.seed(seed)

    xdata_uu = 'experiments/' + exp_name + '/x_uu.dat'
    xdata_dd = 'experiments/' + exp_name + '/x_dd.dat'
    df_uu = read_exp(xdata_uu)
    df_dd = read_exp(xdata_dd)
    q_exp = df_uu['Q'].values

    # approximate noise from errorbars of experimental data (assume errorbar = 1 std)
    dx_uu = (df_uu['dR'].values/df_uu['R'].values)
    dx_dd = (df_dd['dR'].values/df_dd['R'].values)
    dx_uu = np.repeat(np.expand_dims(dx_uu, axis=0), x.shape[0], axis=0)
    dx_dd = np.repeat(np.expand_dims(dx_dd, axis=0), x.shape[0], axis=0)
    x_std = np.stack([dx_uu, dx_dd], axis=2)
    x_noise = x_std*np.random.standard_normal(size=x_std.shape)

    # add noise to simulated data
    x = x*(1. + x_noise)

    # interpolate to simulated q-values
    x = interp1d(q_exp, x, fill_value='extrapolate', axis=1)(q/10.)
    x[x <= 0] = x_orig[x <= 0]

    # apply gentle smoothing
    window_radius = int(len(q)/32.)
    x[:,:,0] = smooth_inverse(q, x[:,:,0], window_radius)
    x[:,:,1] = smooth_inverse(q, x[:,:,1], window_radius)
    return x


def normalize_log(x, x_moms=None):
    z = np.log10(x)
    if x_moms == None:
        zmin = z.ravel().min()
        zmax = z.ravel().max()
    else:
        zmin = x_moms[0]
        zmax = x_moms[1]
    z = (z - zmin)/(zmax - zmin)
    return z, (zmin, zmax)


def normalize_log_inverse(z, x_moms):
    zmin = x_moms[0]
    zmax = x_moms[1]
    z = z*(zmax - zmin) + zmin
    x = 10.**z
    return x


def split_data(x_data, y_data, seed):
    x_train, x_dev, y_train, y_dev = train_test_split(x_data, y_data, test_size=0.3, random_state=seed,
                                                      stratify=y_data[:,-1])
    x_valid, x_test, y_valid, y_test = train_test_split(x_dev, y_dev, test_size=2./3., random_state=seed,
                                                        stratify=y_dev[:,-1])
    print('train size:', x_train.shape[0], 'positive examples:', np.count_nonzero(y_train[:,-1]),
          'negative examples:', len(y_train[:,-1]) - np.count_nonzero(y_train[:,-1]))
    print('valid size:', x_valid.shape[0], 'positive examples:', np.count_nonzero(y_valid[:,-1]),
          'negative examples:', len(y_valid[:,-1]) - np.count_nonzero(y_valid[:,-1]))
    print('test size:', x_test.shape[0], 'positive examples:', np.count_nonzero(y_test[:,-1]),
          'negative examples:', len(y_test[:,-1]) - np.count_nonzero(y_test[:,-1]))
    return x_train, x_valid, x_test, y_train, y_valid, y_test

#################################################################################################################

def smooth_inverse(x, y, window_radius):
    xi = np.linspace(1./x[-1], 1./x[0], 10*len(x))
    f = interp1d(1./x, y, kind='linear', fill_value='extrapolate')
    ys = smooth_data(f(xi), window_radius)
    f = interp1d(1./xi, ys, kind='linear', fill_value='extrapolate')
    return f(x)


def smooth_data(x, window_radius):
    window_len = 2*window_radius+1
    w = np.hanning(window_len)
    
    if len(x.shape) > 1:
        s = np.r_['1', x[:,window_len-1:0:-1], x, x[:,-2:-window_len-1:-1]]
        y = np.zeros((x.shape[0], s.shape[1] - len(w) + 1))
        for i in range(x.shape[0]):
            y[i,:] = np.convolve(s[i,:], w/w.sum(), mode='valid')
        return y[:,window_radius:x.shape[-1] + window_radius]
    
    else:
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        y = np.convolve(s, w/w.sum(), mode='valid')
        return y[window_radius:x.shape[-1] + window_radius]

#################################################################################################################

def get_predictions(model, data_loaders, d_sets, device, height, width, num_features, kwargs, model_name,
                    z_std_norm, y_ids, scaler):
    
    df = pd.DataFrame({'set': [j for i in [[d_set]*len(data_loaders[d_set].dataset) for d_set in d_sets] for j in i]})

    xs, x_preds, x_mses, zs, z_mses, ys, y_preds, y_errs = [], [], [], [], [], [], [], []
    for d_set in d_sets:
        # initialize arrays
        x_pred = np.zeros((len(data_loaders[d_set].dataset), height, width))
        x_mse = np.zeros((len(data_loaders[d_set].dataset),))
        z = np.zeros((len(data_loaders[d_set].dataset), kwargs['latent_dim']))

        if not z_std_norm: z_mse = np.zeros((len(data_loaders[d_set].dataset), num_features))
        else: z_mse = None

        if model_name == 'rcvae':
            y_pred = np.zeros((len(data_loaders[d_set].dataset), num_features + 1))
            y_err = np.zeros((len(data_loaders[d_set].dataset), num_features))

            # predict
            predict(model, data_loaders[d_set], device, y_ids, x_pred, z, x_mse, y_pred, y_err=y_err, z_mse=z_mse)

            # invert standardization
            y_pred[:,:-1] = scaler.inverse_transform(y_pred[:,:-1])
            y_err = scaler.inverse_transform(y_err) - scaler.mean_

        elif model_name == 'rvae':
            y_pred = np.zeros((len(data_loaders[d_set].dataset), num_features))
            y_err = np.zeros((len(data_loaders[d_set].dataset), num_features))

            # predict
            predict(model, data_loaders[d_set], device, y_ids, x_pred, z, x_mse, y_pred, y_err=y_err, z_mse=z_mse)

            # invert standardization
            y_pred = scaler.inverse_transform(y_pred)
            y_err = scaler.inverse_transform(y_err) - scaler.mean_

        elif model_name == 'cvae':
            y_pred = np.zeros((len(data_loaders[d_set].dataset), 1))
            y_err = None

            # predict
            predict(model, data_loaders[d_set], device, y_ids, x_pred, z, x_mse, y_pred, z_mse=z_mse)

        else:
            y_pred = None
            y_err = None

            # predict
            predict(model, data_loaders[d_set], device, y_ids, x_pred, z, x_mse, z_mse=z_mse)

        # invert standardization
        if not z_std_norm:
            z[:,:len(y_ids)-1] = scaler.inverse_transform(z[:,:len(y_ids)-1])
            z_mse[:,:len(y_ids)-1] = scaler.inverse_transform(z_mse[:,:len(y_ids)-1]) - scaler.mean_
        
        x = np.copy(data_loaders[d_set].dataset[:][0].numpy())
        y = np.copy(data_loaders[d_set].dataset[:][1].numpy())
        y[:,y_ids[:-1]] = scaler.inverse_transform(y[:,y_ids[:-1]])
        
        xs += [x]
        ys += [y]
        x_preds += [x_pred]
        zs += [z]
        x_mses += [x_mse]
        z_mses += [z_mse]
        y_preds += [y_pred]
        y_errs += [y_err]
    
    df['x_true'] = [k for k in np.concatenate(xs, axis=0).squeeze()]
    df['x_pred'] = [k for k in np.concatenate(x_preds, axis=0).squeeze()]
    df['x_mse'] = [k for k in np.concatenate(x_mses, axis=0)]
    df['y_true'] = [k for k in np.concatenate(ys, axis=0)]

    try: len(y_preds[0])
    except: pass
    else: df['y_pred'] = [k for k in np.concatenate(y_preds, axis=0)]

    try: len(y_errs[0])
    except: pass
    else: df['y_err'] = [k for k in np.concatenate(y_errs, axis=0)]

    df['z'] = [k for k in np.concatenate(zs, axis=0)]

    try: len(z_mses[0])
    except: pass
    else: df['z_mse'] = [k for k in np.concatenate(z_mses, axis=0)]
    
    return df


def get_predictions_exp(model, x_exp, exp_names, device, height, width, num_features, kwargs, model_name, z_std_norm,
                        y_ids, scaler):
    
    temps = [i[i.index('/')+1:] for i in exp_names]
    df = pd.DataFrame({'set': temps,
                       'x_true': [k for k in x_exp.squeeze(1).numpy()]})
    
    x_exp_pred = np.zeros((x_exp.shape[0], height, width))
    z_exp = np.zeros((x_exp.shape[0], kwargs['latent_dim']))
    x_exp_mse = np.zeros((x_exp.shape[0],))
    
    if model_name == 'rcvae':
        y_exp_pred = np.zeros((x_exp.shape[0], num_features + 1))
        predict_exp(model, x_exp, device, x_exp_pred, z_exp, x_exp_mse, y_exp_pred)
        y_exp_pred[:,:-1] = scaler.inverse_transform(y_exp_pred[:,:-1])

    elif model_name == 'rvae':
        y_exp_pred = np.zeros((x_exp.shape[0], num_features))
        predict_exp(model, x_exp, device, x_exp_pred, z_exp, x_exp_mse, y_exp_pred)
        y_exp_pred = scaler.inverse_transform(y_exp_pred)

    elif model_name == 'cvae':
        y_exp_pred = np.zeros((x_exp.shape[0], 1))
        predict_exp(model, x_exp, device, x_exp_pred, z_exp, x_exp_mse, y_exp_pred)

    else:
        y_exp_pred = None
        predict_exp(model, x_exp, device, x_exp_pred, z_exp, x_exp_mse)

    if not z_std_norm:
        z_exp[:,:len(y_ids)-1] = scaler.inverse_transform(z_exp[:,:len(y_ids)-1])
    
    df['x_pred'] = [k for k in x_exp_pred]
    df['x_mse'] = [k for k in x_exp_mse]
    try: len(y_exp_pred)
    except: pass
    else: df['y_pred'] = [k for k in y_exp_pred]
    df['z'] = [k for k in z_exp]
    
    return df

#################################################################################################################

def format_metric_name(x):
    x = x.replace('_', ' ').capitalize().replace('Recon', 'Reconstruction').replace('Kld', 'KLD').replace('mse', 'MSE')
    return x

def plot_history(image_dir, dynamics, logscale=False):
    prop.set_size(18)
    lprop = prop.copy()
    lprop.set_size(14)

    metrics = dynamics[0]['train'].keys()
    epochs = [d['epoch'] for d in dynamics]
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(5.5,5))
        ax.plot(epochs, [d['train'][metric] for d in dynamics], color='#316986', label='Train')
        ax.plot(epochs, [d['valid'][metric] for d in dynamics], color='#C86646', label='Valid.')

        ax.legend(prop=lprop, frameon=False, loc='best')
        if logscale:
            ax.set_yscale('log')
        else:
            d_min = min([d['train'][metric] for d in dynamics])
            d_max = max([d['train'][metric] for d in dynamics])
            ax.locator_params(axis='y', nbins=5)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            ax.yaxis.offsetText.set_fontproperties(prop)

        format_axis(ax, 'epoch', format_metric_name(metric), prop)
        fig.savefig(image_dir + '/' + metric + '.pdf')


def plot_history_statistics(image_dir, dynamics_list, logscale=False):
    prop.set_size(18)
    lprop = prop.copy()
    lprop.set_size(14)

    metrics = dynamics_list[0][0]['train'].keys()
    epochs = [d['epoch'] for d in dynamics_list[0]]

    # consolidate histories
    dynamics_stat = [{key: {m: [] for m in metrics} for key in ['train', 'valid']} for k in range(len(epochs))]
    for epoch in epochs:
        for metric in metrics:
            for dynamics in dynamics_list:
                dynamics_stat[epoch-1]['train'][metric].append(dynamics[epoch-1]['train'][metric])
                dynamics_stat[epoch-1]['valid'][metric].append(dynamics[epoch-1]['valid'][metric])

            dynamics_stat[epoch-1]['train'][metric] = np.array(dynamics_stat[epoch-1]['train'][metric])
            dynamics_stat[epoch-1]['valid'][metric] = np.array(dynamics_stat[epoch-1]['valid'][metric])
    
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(5.5,5))
        train_avg = np.array([d['train'][metric].mean() for d in dynamics_stat])
        valid_avg = np.array([d['valid'][metric].mean() for d in dynamics_stat])
        train_std = np.array([d['train'][metric].std() for d in dynamics_stat])
        valid_std = np.array([d['valid'][metric].std() for d in dynamics_stat])

        ax.fill_between(epochs, train_avg - train_std, train_avg + train_std, color='#316986', alpha=0.3, lw=0)
        ax.fill_between(epochs, valid_avg - valid_std, valid_avg + valid_std, color='#C86646', alpha=0.3, lw=0)
        ax.plot(epochs, train_avg, color='#316986', label='Train')
        ax.plot(epochs, valid_avg, color='#C86646', label='Valid.')

        ax.legend(prop=lprop, frameon=False, loc='best')
        if logscale:
            ax.set_yscale('log')
        else:
            ax.locator_params(axis='y', nbins=5)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            ax.yaxis.offsetText.set_fontproperties(prop)
        
        format_axis(ax, 'epoch', format_metric_name(metric), prop)
        fig.savefig(image_dir + '/' + metric + '_stats.pdf')

#################################################################################################################

def plot_decoded(image_dir, x_pred, x_true, x_mse, d_set, x_exp_mse, exp_names):
    n = 6
    fig = plt.figure(figsize=(2*n,7))
    tprop = prop.copy()
    tprop.set_size(12)
    
    # identify quartiles by MSE
    i_mse = np.argsort(x_mse)
    x_mse = x_mse[i_mse]
    x_pred = x_pred[i_mse]
    x_true = x_true[i_mse]
    quartiles = np.quantile(x_mse, (0.25, 0.5, 0.75, 1.))
    i_quart = [0] + [np.argmin(np.abs(x_mse - k)) for k in quartiles]
    idx = np.concatenate([np.sort(np.random.randint(i_quart[k-1], i_quart[k], size=n)) for k in range(1,5)])

    for k in range(4*n):
        ax = fig.add_subplot(4,n,k+1)
        i = idx[k]
        ax.plot(range(x_pred.shape[1]), x_pred[i,:,0], color='#316986', alpha=0.7)
        ax.plot(range(x_pred.shape[1]), x_pred[i,:,1], color='#C86646', alpha=0.7)
        ax.plot(range(x_true.shape[1]), x_true[i,:,0], color='#316986')
        ax.plot(range(x_true.shape[1]), x_true[i,:,1], color='#C86646')
        ax.text(0.9, 0.9, 'MSE', ha='right', va='center', transform=ax.transAxes, fontproperties=tprop)
        ax.text(0.9, 0.78, '{:.2e}'.format(x_mse[i]), ha='right', va='center', transform=ax.transAxes,
                fontproperties=tprop)
        
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis('off')
        
    fig.tight_layout()
    fig.savefig(image_dir + '/decoded_' + d_set + '.pdf')

    # plot quartile distribution
    fig, ax = plt.subplots(figsize=(2.5,7))
    prop.set_size(16)
    y_min, y_max = x_mse.min(), x_mse.max()
    n = 1000
    yy = np.linspace(y_min, y_max, n)
    kde = gaussian_kde(x_mse)
    py = kde.pdf(yy)
    ax.plot(py, yy, color='black')
    qs = (0.25, 0.5, 0.75)
    quartiles = np.quantile(x_mse, qs)
    for i, q in enumerate(quartiles):
        ax.axhline(q, linestyle='--', color='gray')
        ax.text(1.1*py.max(), q, '{:.2f}'.format(qs[i]), color='gray', fontproperties=tprop, ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='white', pad=0.05))

    format_axis(ax, '', 'MSE', prop, nbins=3)
    ax.set_ylim([y_min, y_max])
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    ax.xaxis.offsetText.set_fontproperties(prop)
    ax.yaxis.offsetText.set_fontproperties(prop)

    temps = [int(i[i.index('/')+1:-1]) for i in exp_names]
    if len(temps) > 1:
        norm = mpl.colors.LogNorm(vmin=min(temps), vmax=max(temps))
    else:
        norm = mpl.colors.LogNorm(vmin=1, vmax=max(temps))
    ax.scatter([0.05*py.max()]*len(x_exp_mse), x_exp_mse, c=cmap_temp(norm(temps)), s=48, ec='black')
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(image_dir + '/decoded_' + d_set + '_quartile.pdf')

def plot_decoded_exp(image_dir, x_pred, exp_names, q, x_moms):
    temps = [int(i[i.index('/')+1:-1]) for i in exp_names]
    norm = mpl.colors.LogNorm(vmin=min(temps), vmax=max(temps))

    prop.set_size(18)
    prop_text = prop.copy()
    prop_text.set_size(14)

    if len(temps) > 1:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9.5,5), sharey=True)

        for i in range(x_pred.shape[0]):
            xdata_uu = 'experiments/' + exp_names[i] + '/x_uu.dat'
            xdata_dd = 'experiments/' + exp_names[i] + '/x_dd.dat'
            df_uu = read_exp(xdata_uu)
            df_dd = read_exp(xdata_dd)
            
            ax1.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,0], x_moms), color=cmap_temp(norm(temps[i])))
            ax2.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,1], x_moms), color=cmap_temp(norm(temps[i])))
            

            ax1.errorbar(10*df_uu['Q'].values, 10**i*df_uu['R'].values, yerr=df_uu['dR'].values, lw=0, elinewidth=1,
                         color=cmap_temp(norm(temps[i])))
            ax2.errorbar(10*df_dd['Q'].values, 10**i*df_dd['R'].values, yerr=df_dd['dR'].values, lw=0, elinewidth=1,
                         color=cmap_temp(norm(temps[i])))
            ax1.scatter(10*df_uu['Q'].values, 10**i*df_uu['R'].values, s=24, color=cmap_temp(norm(temps[i])),
                        ec='none', label=str(temps[i]) + 'K')
            ax2.scatter(10*df_dd['Q'].values, 10**i*df_dd['R'].values, s=24, color=cmap_temp(norm(temps[i])),
                        ec='none', label=str(temps[i]) + 'K')

        ax1.legend(frameon=False, ncol=2, prop=prop_text, loc='upper right', columnspacing=1.4, handletextpad=0.4)
        ax2.legend(frameon=False, ncol=2, prop=prop_text, loc='upper right', columnspacing=1.4, handletextpad=0.4)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        format_axis(ax1, 'Q (nm$^{-1}$)', 'Reflectivity', prop, 'R$^{++}$', [q.min(), q.max()])
        format_axis(ax2, 'Q (nm$^{-1}$)', ' ', prop, 'R$^{--}$', [q.min(), q.max()])

    else:
        fig, ax = plt.subplots(1,1, figsize=(5.5,5))

        for i in range(x_pred.shape[0]):
            xdata_uu = 'experiments/' + exp_names[i] + '/x_uu.dat'
            xdata_dd = 'experiments/' + exp_names[i] + '/x_dd.dat'
            df_uu = read_exp(xdata_uu)
            df_dd = read_exp(xdata_dd)
            
            p1, = ax.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,0], x_moms), color='#316986')
            p2, = ax.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,1], x_moms), color='#C86646')
            

            ax.errorbar(10*df_uu['Q'].values, 10**i*df_uu['R'].values, yerr=df_uu['dR'].values, lw=0, elinewidth=1,
                        color='#316986')
            s1 = ax.scatter(10*df_uu['Q'].values, 10**i*df_uu['R'].values, s=24, color='#316986',
                            ec='none', label='R$^{++}$')
            ax.errorbar(10*df_dd['Q'].values, 10**i*df_dd['R'].values, yerr=df_dd['dR'].values, lw=0, elinewidth=1,
                        color='#C86646')
            s2 = ax.scatter(10*df_dd['Q'].values, 10**i*df_dd['R'].values, s=24, color='#C86646',
                            ec='none', label='R$^{--}$')

        ax.legend([s1, p1, s2, p2], ['R$^{++}$', 'Pred.', 'R$^{--}$', 'Pred.'], frameon=False, ncol=2, prop=prop_text,
                  loc='lower left', columnspacing=1)
        ax.set_yscale('log')
        format_axis(ax, 'Q (nm$^{-1}$)', 'Reflectivity', prop, '', [q.min(), q.max()])


        # zoom inset
        fig_, ax_ = plt.subplots(1,1, figsize=(5.5/1.5,5/2.))

        for i in range(x_pred.shape[0]):
            xdata_uu = 'experiments/' + exp_names[i] + '/x_uu.dat'
            xdata_dd = 'experiments/' + exp_names[i] + '/x_dd.dat'
            df_uu = read_exp(xdata_uu)
            df_dd = read_exp(xdata_dd)
            
            p1, = ax_.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,0], x_moms), color='#316986')
            p2, = ax_.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,1], x_moms), color='#C86646')
            

            ax_.errorbar(10*df_uu['Q'].values, 10**i*df_uu['R'].values, yerr=df_uu['dR'].values, lw=0, elinewidth=1,
                         color='#316986')
            s1 = ax_.scatter(10*df_uu['Q'].values, 10**i*df_uu['R'].values, s=24, color='#316986',
                             ec='none', label='R$^{++}$')
            ax_.errorbar(10*df_dd['Q'].values, 10**i*df_dd['R'].values, yerr=df_dd['dR'].values, lw=0, elinewidth=1,
                         color='#C86646')
            s2 = ax_.scatter(10*df_dd['Q'].values, 10**i*df_dd['R'].values, s=24, color='#C86646',
                             ec='none', label='R$^{--}$')

        ax_.set_yscale('log')
        format_axis(ax_, 'Q (nm$^{-1}$)', 'Refl.', prop, '', [1, q.max()], [1e-6, 5e-5])

        fig_.tight_layout()
        fig_.subplots_adjust(wspace=0.1)
        fig_.savefig(image_dir + '/decoded_exp_inset.pdf')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(image_dir + '/decoded_exp.pdf')

#################################################################################################################

def plot_predicted(image_dir, y_pred, y_true, y_ids, y_labels, y_units):
    prop.set_size(14)
    if len(y_ids) > 15: n = 5
    else: n = 3
    w = int((len(y_ids) + (len(y_ids)%n > 0)*(n - len(y_ids)%n))/n)
    fig = plt.figure(figsize=(3.3*w, n*2.67))
    
    for k in range(len(y_ids)-1):
        ax = fig.add_subplot(n, w, k+1)
        
        bins = 60
        norm = mpl.colors.LogNorm(vmin=1, vmax=1e2)
        
        _, _, _, g = ax.hist2d(y_true[:,y_ids[k]], y_pred[:,k], bins=bins, cmap=cmap_mse, norm=norm, cmin=1, alpha=0.8,
                               ec='none')
        cbar = fig.colorbar(g, ax=ax, aspect=12)
        cbar.ax.tick_params(direction='in', length=6, width=1)
        cbar.ax.tick_params(which='minor', length=0)
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontproperties(prop)
        cbar.outline.set_visible(False)
        
        format_axis(ax, y_labels[k] + ' ' + y_units[k], y_labels[k] + ' ' + y_units[k], prop)
        ax.locator_params(tight=True, nbins=4)
        ax.set_aspect('equal')
        
        x_min = y_true[:,y_ids[k]].min() - 0.05
        x_max = y_true[:,y_ids[k]].max() + 0.05
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([x_min, x_max])
        ax.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--')
        
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    fig.savefig(image_dir + '/predicted.png', dpi=400)


def plot_predicted_property(image_dir, y_pred, y_true, y_err, y_name, y_header, y_ids, y_labels, y_units,
                            qs=(0.5, 0.75, 0.95)):
    prop.set_size(14)
    tprop = prop.copy()
    tprop.set_size(12)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(1.2*4.8, 1.2*2.67), gridspec_kw=dict(width_ratios=[2,1]))
    
    k = y_header.index(y_name)
    
    # predicted vs. true values
    bins = 60
    norm = mpl.colors.LogNorm(vmin=1, vmax=1e2)
    
    _, _, _, g = ax1.hist2d(y_true[:,y_ids[k]], y_pred[:,k], bins=bins, cmap=cmap_mse, norm=norm, cmin=1, alpha=0.8,
                           ec='none')

    ax_ = inset_axes(ax1, width='100%', height='8%', loc='upper center', borderpad=-2)
    cbar = fig.colorbar(g, cax=ax_, orientation='horizontal')
    cbar.ax.tick_params(direction='in', length=6, width=1)
    cbar.ax.tick_params(which='minor', length=0)
    cbar.ax.xaxis.set_ticks_position('top')
    for lab in cbar.ax.get_xticklabels():
        lab.set_fontproperties(prop)
    cbar.outline.set_visible(False)
        
    format_axis(ax1, 'True ' + y_labels[k] + ' ' + y_units[k], 'Pred. ' + y_labels[k] + ' ' + y_units[k], prop)
    ax1.locator_params(tight=True, nbins=4)
    ax1.set_aspect('equal')
    
    x_min = y_true[:,y_ids[k]].min() - 0.05
    x_max = y_true[:,y_ids[k]].max() + 0.05
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([x_min, x_max])
    ax1.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--')
    
    # error distribution
    y_min, y_max = y_err[:,k].min(), y_err[:,k].max()
    n = 1000
    yy = np.linspace(y_min, y_max, n)
    kde = gaussian_kde(y_err[:,k])
    py = kde.pdf(yy)
    ax2.plot(py, yy, color='black')

    # plot quartiles
    if len(qs) < 4: i0 = 0
    else: i0 = 2
    quartiles = np.quantile(y_err[:,k], qs)
    for i, q in enumerate(quartiles):
        ax2.axhline(q, linestyle='--', color='gray')
        if i > i0: p = 0.72*py.max()
        else: p = 0.05*py.max()
        ax2.text(p, q, '{:.2f}'.format(qs[i]), color='gray', fontproperties=tprop, ha='left', va='center',
                 bbox=dict(facecolor='white', edgecolor='white', pad=0.05))

    format_axis(ax2, 'pdf', r'$L_1$ ' + y_units[k], prop, nbins=3)
    ax2.set_ylim([y_min, y_max])

    fig.subplots_adjust(wspace=0.35, top=0.8, bottom=0.15)
    fig.savefig(image_dir + '/predicted_' + y_name + '.png', dpi=400)


def get_roc(df, d_sets):
    fpr = []; tpr = []; roc_auc = []; th = []
    for d_set in d_sets:
        y_true = np.stack(df.loc[df['set']==d_set, 'y_true'].values)[:,-1]
        y_pred = np.stack(df.loc[df['set']==d_set, 'y_pred'].values)[:,-1]

        fpri, tpri, thi = roc_curve(y_true, y_pred)
        fpr += [fpri]; tpr += [tpri]; th += [thi]
        roc_auc += [auc(fpri, tpri)]
    return fpr, tpr, roc_auc, th


def get_roc_prox(df, y_header, column):
    y_true = np.stack(df.loc[df['set']=='valid', 'y_true'].values)[:,-1]
    y_pred = np.stack(df.loc[df['set']=='valid', 'y_pred'].values)[:,y_header.index(column)]

    fpr, tpr, th = roc_curve(y_true, y_pred)
    return fpr, tpr, th


def get_optimal_threshold(fpr, tpr, th):
    k = np.argmin(np.abs(tpr - (1 - fpr)))
    print('tpr:', tpr[k], 'fpr:', fpr[k], 'threshold:', th[k])
    return tpr[k], fpr[k], th[k]


def get_precision_recall_f1(df, d_set, th=0.5):
    y_true = np.stack(df.loc[df['set']==d_set, 'y_true'].values)[:,-1]
    y_pred = np.stack(df.loc[df['set']==d_set, 'y_pred'].values)[:,-1]
    y_pred = (y_pred >= th).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    return precision, recall, f1


def plot_roc(image_dir, fpr, tpr, roc_auc, op=None):
    prop.set_size(18)
    tprop = prop.copy()
    tprop.set_size(16)
    fig, ax = plt.subplots(figsize=(5.5,5))

    if op: ax.scatter(op[1], op[0], s=48, facecolor='white', edgecolor='black',
                      lw=1.5, zorder=10, label=r'$t_{cutoff}$:' + ' %0.2f' % op[2])

    for i, (key, color) in enumerate(set_colors.items()):
        ax.plot(fpr[i], tpr[i], color=color, label=r'AUC$_{' + key + r'}$:' + ' %0.2f' % roc_auc[i])
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    format_axis(ax, 'false positive rate', 'true positive rate', prop, xlims=[0,1], ylims=[0,1.05])
    ax.legend(frameon=False, loc='lower right', prop=tprop)

    fig.tight_layout()
    fig.savefig(image_dir + '/roc.pdf')


def plot_precision_recall_f1(image_dir, df, d_sets, th=0.5):
    prop.set_size(16)
    tprop = prop.copy()
    tprop.set_size(14)
    bx = np.arange(3)
    width = 0.25
    sep = 0.18
    scores_dict = {'precision': 0, 'recall': 1, 'f1': 2}

    for d_set in d_sets:
        fig, ax = plt.subplots(figsize=(5,3.2))

        scores = get_precision_recall_f1(df, d_set, th)
        y0 = [scores[scores_dict[score]][0] for score in ['recall', 'precision', 'f1']]
        y1 = [scores[scores_dict[score]][1] for score in ['recall', 'precision', 'f1']]

        ax.bar(bx - sep, y0, width, color=cmap_disc(0), label='no proximity')
        ax.bar(bx + sep, y1, width, color=cmap_disc(1), label='with proximity')

        for i in range(len(bx)):
            ax.text(bx[i] - sep, y0[i] + 0.03, '%0.2f' % y0[i], color=cmap_disc(0), ha='center', fontproperties=tprop)
            ax.text(bx[i] + sep, y1[i] + 0.03, '%0.2f' % y1[i], color=cmap_disc(1), ha='center', fontproperties=tprop)

        format_axis(ax, 'metric', 'score', prop, ylims=[0,1.1])
        ax.set_xticks(bx)
        ax.set_xticklabels(['recall', 'precision', r'$F_1$'])
        ax.legend(frameon=False, bbox_to_anchor=(0., 1., 1., .102), prop=tprop, loc='lower left', ncol=2,
                  mode='expand', borderaxespad=0.)

        fig.tight_layout()
        fig.savefig(image_dir + '/scores_' + d_set + '.pdf', dpi='figure')


def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    return np.round(cm, 2)


def plot_confusion_matrix(image_dir, df, y_header, y_name, y_th):
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cm = get_confusion_matrix(np.stack(df.loc[df['set']=='test', 'y_true'])[:,-1].astype(int),
                             (np.stack(df.loc[df['set']=='test', 'y_pred'])[:,y_header.index(y_name)]>=y_th).astype(int))

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(2.3,2.3))
    prop.set_size(14)
    squares = [mpatch.Rectangle((-1,0), 1, 1, color=cmap_cm(norm(cm[0,0]))),
               mpatch.Rectangle((0,0), 1, 1, color=cmap_cm(norm(cm[0,1]))),
               mpatch.Rectangle((-1,-1), 1, 1, color=cmap_cm(norm(cm[1,0]))),
               mpatch.Rectangle((0,-1), 1, 1, color=cmap_cm(norm(cm[1,1])))]

    for (square, value) in zip(squares, cm.ravel()):
        ax.add_artist(square)
        rx, ry = square.get_xy()
        cx = rx + square.get_width()/2.0
        cy = ry + square.get_height()/2.0

        if value >= 0.7: color = 'white'
        else: color='black'
        ax.annotate('{:.2f}'.format(value), (cx, cy), color=color, fontproperties=prop, ha='center', va='center')

    format_axis(ax, 'Pred.', 'True', prop)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(image_dir + '/confusion_matrix_' + y_name + '.pdf', bbox_inches='tight')
    
    # save colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap_cm, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(0.21,2.5))
    cbar = fig.colorbar(sm, cax=ax, ticks=[0, 0.5, 1])
    format_axis(cbar.ax, '', '', prop)
    cbar.ax.tick_params(which='major', length=0, width=0)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontproperties(prop)
    cbar.outline.set_visible(False)
    fig.savefig(image_dir + '/confusion_matrix_' + y_name + '_cbar.pdf', bbox_inches='tight')

#################################################################################################################     

def gradient_pt_cloud(x, y):
    # Adapted from: https://stackoverflow.com/a/40520133
    
    # number of nearest neighbors to include
    k = 32
    kdtree = KDTree(x)
    
    Dy = np.zeros(x.shape)
    for i in range(x.shape[0]):
        # get coordinates of k nearest neighbors
        x0 = x[[i],:]
        y0 = y[i]
        _, idx = kdtree.query(x0, k)
        idx = idx[0]
        X = x[idx,:]
        Y = y[idx]
        
        # calculate differences from central node
        X0 = (np.dot(x0.T, np.ones((1,k)))).T
        dX = X - X0
        dY = Y - y0
        
        # calculate coefficients (a)
        G = np.dot(dX, dX.T)
        F = np.multiply(G, G)
        v = np.diag(G)
        N = np.dot(np.linalg.pinv(G), G)
        N = np.eye(k) - N
        a = np.dot(np.dot(N, np.linalg.pinv(np.dot(F, N))), v)
        
        # estimate gradient
        Dy[i,:] = np.dot(dX.T, np.multiply(a, dY))
        
    return Dy


def plot_latent_representation(image_dir, x_data, y_data, y_ids, y_labels, y_units, tag='', z_exp=None, exp_names=None,
                               mode=2):
    # visualize separability by y-values
    mode_dict = {0: 'x-pca', 1: 'y-pca', 2: 'y-grad', 3: 'x-mds'}

    prop.set_size(14)
    if len(y_ids) > 15: n = 5
    else: n = 3
    w = int((len(y_ids) + (len(y_ids)%n > 0)*(n - len(y_ids)%n))/n)
    fig = plt.figure(figsize=(3*w, n*2.5))

    # downsampling
    if mode < 3:
        d = max(1, len(x_data)//5000)
        alpha = 0.2
    else:
        d = max(1, len(x_data)//500)
        alpha = 0.5
    
    try: len(z_exp)
    except: pass
    else:
        n_exp = z_exp.shape[0]
        exp_names = [i[i.index('/')+1:] for i in exp_names]
        temps = [int(k[:-1]) for k in exp_names]
        if len(temps) > 1:
            norm = mpl.colors.LogNorm(vmin=min(temps), vmax=max(temps))
        else:
            norm = mpl.colors.LogNorm(vmin=1, vmax=max(temps))
    
    xd = x_data[::d,:]
    for k in range(len(y_ids)):
        ax = fig.add_subplot(n, w, k+1)
        yd = y_data[::d,y_ids[k]]
        
        if mode == 0:
            pca = PCA(n_components=2)
            x_proj = pca.fit_transform(xd)

        elif mode == 1:
            pca = PCA(n_components=2)
            pca.fit(np.repeat(np.expand_dims(yd, 1), xd.shape[1], axis=1))
            x_proj = pca.transform(xd)

        elif mode == 2:
            Dy = gradient_pt_cloud(xd, yd)
            idx = np.argsort(Dy.mean(axis=0))[::-1]
            x_proj = np.concatenate([xd[:,[idx[0]]], xd[:,[idx[1]]]], axis=1)
        
        else:
            mds = MDS(n_components=2)
            x_proj = mds.fit_transform(xd)
            
        if 'class' in y_labels[k]:
            g = ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, cmap=cmap_disc)
            cbar = fig.colorbar(g, ax=ax, aspect=12, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['0', '+'])
        else:
            g = ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, cmap=cmap)
            cbar = fig.colorbar(g, ax=ax, aspect=12)

        cbar.ax.set_title(y_units[k], fontproperties=prop)
        cbar.ax.tick_params(direction='in', length=6, width=1)
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontproperties(prop)
        cbar.outline.set_visible(False)
        ax.clear()

        if 'class' in y_labels[k]: ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, alpha=alpha, cmap=cmap_disc)
        else: ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, alpha=alpha, cmap=cmap)
        
        try: len(z_exp)
        except: pass
        else:
            if mode < 2: z_proj = pca.transform(z_exp)
            elif mode == 2: z_proj = np.concatenate([z_exp[:,[idx[0]]], z_exp[:,[idx[1]]]], axis=1)
            else: z_proj = mds.transform(z_exp)

            for i in range(n_exp):
                ax.scatter(z_proj[[i],0], z_proj[[i],1], color=cmap_temp(norm(temps[i])), ec='black', s=36)
                
        ax.text(0.075, 0.9, y_labels[k], ha='left', va='center', transform=ax.transAxes, fontproperties=prop)

        format_axis(ax, '', '', prop)
        ax.locator_params(tight=True, nbins=4)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    fig.savefig(image_dir + '/' + tag + '_' + mode_dict[mode] + '.png', dpi=400)

#################################################################################################################  

def plot_class_exp_statistics(image_dir, df_exp, reps):
    # plot statistics of proximity class
    df_exp['temperature'] = df_exp['set'].map(lambda x: int(x[:-1]))
    num_temps = len(np.unique(df_exp['temperature']))

    df_exp['class_pred'] = df_exp['y_pred'].map(lambda x: x[-1])
    df_exp['class_dist'] = df_exp['class_pred'] - df_exp['th']
    df_exp['class_prox'] = (df_exp['class_dist'] > 0).astype(int)
    df_exp['major_class'] = 0
    
    for entry in df_exp.itertuples():
        if entry.class_prox > 0:
            df_exp.loc[df_exp['set'] == entry.set, 'major_class'] += 1
    df_exp['major_class'] = df_exp['major_class']/reps
    df_exp['major_class'] = (df_exp['major_class'] >= 0.5).astype(int)
    df_exp['palette'] = df_exp['major_class'].map(lambda x: cmap_disc_light(x))

    fig, ax = plt.subplots(figsize=(0.25 + num_temps, 3.2))
    prop.set_size(14)

    sns.violinplot(ax=ax, x='temperature', y='class_dist', width=0.6, scale='count', data=df_exp,
                   inner=None, saturation=0.9, palette=df_exp['palette'].tolist(), linewidth=0)
    g = sns.stripplot(ax=ax, x='temperature', y='class_dist', hue='class_prox', data=df_exp, jitter=0.05,
                      palette={0: cmap_disc(0), 1: cmap_disc(1)}, edgecolor='black', linewidth=1)
    g.legend_.remove()
    ax.axhline(0, linestyle='--', color='#ADABA4')
    
    if num_temps > 1:
        format_axis(ax, 'Temperature (K)', r'$class_{pred} - t_{cutoff}$', prop)
    else:
        T = np.unique(df_exp['temperature'])[0]
        format_axis(ax, 'T (K)', r'$class_{pred} - t_{cutoff}$', prop)

    fig.subplots_adjust(bottom=0.2)
    fig.savefig(image_dir + '/class_distribution_exp.pdf')


def plot_exp_statistics(image_dir, df_exp, reps, y_name, y_header, y_labels, y_units, y_th=None, y_lims=None):
    # plot statistics of parameter in given column of latent representation z
    column = y_header.index(y_name)
    df_exp['temperature'] = df_exp['set'].map(lambda x: int(x[:-1]))

    if 'y_pred' in df_exp.columns:
        df_exp['p'] = df_exp['y_pred'].map(lambda x: x[column])
    else:
        df_exp['p'] = df_exp['z'].map(lambda x: x[column])
    
    if y_th != None:
        df_exp['p_dist'] = df_exp['p'] - y_th
        df_exp['p_class'] = (df_exp['p_dist'] > 0).astype(int)
        df_exp['major_class'] = 0
        for entry in df_exp.itertuples():
            if entry.p_class > 0:
                df_exp.loc[df_exp['set'] == entry.set, 'major_class'] += 1
        df_exp['major_class'] = df_exp['major_class']/reps
        df_exp['major_class'] = (df_exp['major_class'] >= 0.5).astype(int)
        df_exp['palette'] = df_exp['major_class'].map(lambda x: cmap_disc_light(x))

    else:
        df_exp['palette'] = '#6A96A9'

    num_temps = len(np.unique(df_exp['temperature']))
    fig, ax = plt.subplots(figsize=(0.75 + 5.25/6.*num_temps, 3.2))
    prop.set_size(14)

    sns.violinplot(ax=ax, x='temperature', y='p', width=0.6, scale='count', data=df_exp,
                   inner=None, saturation=0.9, palette=df_exp['palette'].tolist(), linewidth=0)

    if y_th != None:
        g = sns.stripplot(ax=ax, x='temperature', y='p', hue='p_class', data=df_exp, jitter=0.05,
                          palette={0: cmap_disc(0), 1: cmap_disc(1)}, edgecolor='black', linewidth=1)
        g.legend_.remove()
        ax.axhline(y_th, linestyle='--', color='#ADABA4')
    else:
        sns.swarmplot(ax=ax, x='temperature', y='p', data=df_exp, color='#316986', edgecolor='black', linewidth=1)

    ylims = ax.get_ylim()
    y_lims = [min(ylims[0], y_lims[0]), max(ylims[1], y_lims[1])]
    if num_temps > 1:
        format_axis(ax, 'Temperature (K)', y_labels[column] + ' ' + y_units[column], prop, ylims=y_lims)
        fig.subplots_adjust(bottom=0.2)

    else:
        T = np.unique(df_exp['temperature'])[0]
        format_axis(ax, 'T (K)', y_labels[column] + ' ' + y_units[column], prop, ylims=y_lims)
        fig.subplots_adjust(bottom=0.2, left=0.45)

    fig.savefig(image_dir + '/' + y_header[column] + '_distribution_exp.pdf')
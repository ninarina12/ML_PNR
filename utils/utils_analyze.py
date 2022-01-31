import numpy as np
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from utils.utils_data import read_exp, normalize_log_inverse
from utils.utils_plot import *


def format_metric_name(x):
	x = x.replace('_', ' ').capitalize().replace('Recon', 'Reconstruction').replace('Kld', 'KLD').replace('mse', 'MSE')
	return x


def plot_history(image_dir, dynamics, logscale=False):
	metrics = dynamics[0]['train'].keys()
	epochs = [d['epoch'] for d in dynamics]
	for i, metric in enumerate(metrics):
		fig, ax = plt.subplots(figsize=(4,3.5))
		ax.plot(epochs, [d['train'][metric] for d in dynamics], lw=1.5, color=set_colors['train'], label='Train')
		ax.plot(epochs, [d['valid'][metric] for d in dynamics], lw=1.5, color=set_colors['valid'], label='Valid.')

		ax.legend(frameon=False, loc='best')
		if logscale: ax.set_yscale('log')
		else:
			d_min = min([d['train'][metric] for d in dynamics])
			d_max = max([d['train'][metric] for d in dynamics])
			ax.locator_params(axis='y', nbins=5)
			ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

		format_axis(ax, 'Epochs', format_metric_name(metric))
		fig.savefig(image_dir + '/' + metric + '.pdf', bbox_inches='tight')


def plot_history_statistics(image_dir, dynamics_list, logscale=False):
	metrics = list(dynamics_list[0][0]['train'].keys())
	epochs = [d['epoch'] for d in dynamics_list[0]]

	# consolidate histories
	dynamics_stat = [{key: {m: [] for m in metrics} for key in ['train', 'valid']} for k in range(len(epochs))]
	for i, epoch in enumerate(epochs):
		for metric in metrics:
			for dynamics in dynamics_list:
				dynamics_stat[i]['train'][metric].append(dynamics[i]['train'][metric])
				dynamics_stat[i]['valid'][metric].append(dynamics[i]['valid'][metric])

			dynamics_stat[i]['train'][metric] = np.array(dynamics_stat[i]['train'][metric])
			dynamics_stat[i]['valid'][metric] = np.array(dynamics_stat[i]['valid'][metric])

	for i, metric in enumerate(metrics):
		fig, ax = plt.subplots(figsize=(4,3.5))
		train_avg = np.array([d['train'][metric].mean() for d in dynamics_stat])
		valid_avg = np.array([d['valid'][metric].mean() for d in dynamics_stat])
		train_std = np.array([d['train'][metric].std() for d in dynamics_stat])
		valid_std = np.array([d['valid'][metric].std() for d in dynamics_stat])

		ax.fill_between(epochs, train_avg - train_std, train_avg + train_std, color=set_colors['train'], alpha=0.3, lw=0)
		ax.fill_between(epochs, valid_avg - valid_std, valid_avg + valid_std, color=set_colors['valid'], alpha=0.3, lw=0)
		ax.plot(epochs, train_avg, color=set_colors['train'], label='Train')
		ax.plot(epochs, valid_avg, color=set_colors['valid'], label='Valid.')

		ax.legend(frameon=False, loc='best')
		if logscale: ax.set_yscale('log')
		else:
			ax.locator_params(axis='y', nbins=5)
			ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

		format_axis(ax, 'Epochs', format_metric_name(metric))
		fig.savefig(image_dir + '/' + metric + '_stats.pdf', bbox_inches='tight')


def plot_weights(model):
	W = model.encoder_conv2d[0].weight.detach().cpu().squeeze().numpy()
	fig, ax = plt.subplots(W.shape[0]//4,4, figsize=(14,W.shape[0]//4), sharex=True, sharey=True)
	ax = ax.ravel()
	norm = plt.Normalize(vmin=-np.abs(W).max(), vmax=np.abs(W).max())
	for i in range(W.shape[0]):
		ax[i].imshow(W[i].T, cmap=cmap_temp)
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	fig.subplots_adjust(wspace=0.01)


def spin_asymmetry(I):
	return (I[0]-I[1])/(I[0]+I[1])


def spin_asymmetry_error(I, dI):
	s0 = 1./(I[0]+I[1]) - (I[0]-I[1])/(I[0]+I[1])**2
	s1 = -1./(I[0]+I[1]) - (I[0]-I[1])/(I[0]+I[1])**2
	return np.sqrt(s0**2*dI[0]**2 + s1**2*dI[1]**2)


def plot_decoded(image_dir, x_pred, x_true, x_mse, d_set, x_exp_mse, exp_names):
	n = 6
	fig = plt.figure(figsize=(2*n,7))
	
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
		ax.plot(range(x_pred.shape[1]), x_pred[i,:,0], color=spin_colors['+'])
		ax.plot(range(x_pred.shape[1]), x_pred[i,:,1], color=spin_colors['-'])
		ax.plot(range(x_true.shape[1]), x_true[i,:,0], color=spin_colors['+'], lw=1.5, alpha=0.7)
		ax.plot(range(x_true.shape[1]), x_true[i,:,1], color=spin_colors['-'], lw=1.5, alpha=0.7)
		ax.text(0.9, 0.9, 'MSE', ha='right', va='center', transform=ax.transAxes, fontsize=textsize-2)
		ax.text(0.9, 0.78, '{:.2e}'.format(x_mse[i]), ha='right', va='center', transform=ax.transAxes, fontsize=textsize-2)
		
		ax.set_xticks([]); ax.set_yticks([])
		ax.axis('off')
		
	fig.tight_layout()
	fig.savefig(image_dir + '/decoded_' + d_set + '.pdf', bbox_inches='tight')

	# plot quartile distribution
	fig, ax = plt.subplots(figsize=(2.5,7))
	y_min, y_max = x_mse.min(), x_mse.mean() + 4*x_mse.std()
	n = 1000
	yy = np.linspace(y_min, y_max, n)
	kde = gaussian_kde(x_mse)
	py = kde.pdf(yy)
	ax.plot(py, yy, color='black')
	qs = (0.25, 0.5, 0.75)
	quartiles = np.quantile(x_mse, qs)
	for i, q in enumerate(quartiles):
		ax.axhline(q, linestyle='--', color='gray')
		ax.text(1.1*py.max(), q, '{:.2f}'.format(qs[i]), color='gray', ha='left', va='center', fontsize=textsize-2,
				bbox=dict(facecolor='white', edgecolor='white', pad=0.05))

	format_axis(ax, '', 'MSE', nbins=3)
	ax.set_ylim([y_min, y_max])
	ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))

	temps = [int(k[:-1]) for k in exp_names]
	if len(temps) > 1: norm = mpl.colors.Normalize(vmin=0, vmax=len(temps)-1)
	else: norm = mpl.colors.Normalize(vmin=0, vmax=1)
	ax.scatter([0.2*py.max()]*len(x_exp_mse), x_exp_mse, c=cmap_temp(norm(list(range(len(temps))))), s=48, ec='black')
	ax.invert_yaxis()

	fig.tight_layout()
	fig.savefig(image_dir + '/decoded_' + d_set + '_quartile.pdf', bbox_inches='tight')


def plot_decoded_exp(image_dir, x_pred, sample_name, exp_names, q, x_moms):
	temps = [int(k[:-1]) for k in exp_names]
	if len(temps) > 1: norm = mpl.colors.Normalize(vmin=0, vmax=len(temps)-1)
	else: norm = mpl.colors.Normalize(vmin=0, vmax=1)

	if len(temps) > 1:
		fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8.5,4.6), sharey=True)
		fig_, ax = plt.subplots(1,1, figsize=(5,4.4))

		for i in range(x_pred.shape[0]):
			xdata_uu = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_uu.dat'
			xdata_dd = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_dd.dat'
			df_uu = read_exp(xdata_uu)
			df_dd = read_exp(xdata_dd)
			
			ax1.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,0], x_moms), color=cmap_temp(norm(i)))
			ax2.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,1], x_moms), color=cmap_temp(norm(i)))
			
			ax1.errorbar(10*df_uu['Q'], 10**i*df_uu['R'], yerr=10**i*df_uu['dR'], lw=0, elinewidth=1, color=cmap_temp(norm(i)))
			ax2.errorbar(10*df_dd['Q'], 10**i*df_dd['R'], yerr=10**i*df_dd['dR'], lw=0, elinewidth=1, color=cmap_temp(norm(i)))
			ax1.scatter(10*df_uu['Q'], 10**i*df_uu['R'], s=24, color=cmap_temp(norm(i)), ec='none', label=str(temps[i]) + 'K')
			ax2.scatter(10*df_dd['Q'], 10**i*df_dd['R'], s=24, color=cmap_temp(norm(i)), ec='none', label=str(temps[i]) + 'K')

			# plot spin channels together for base temperature
			if i == 0:
				p1, = ax.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,0], x_moms), color=spin_colors['+'])
				p2, = ax.plot(q, 10**i*normalize_log_inverse(x_pred[i,:,1], x_moms), color=spin_colors['-'])
				
				ax.errorbar(10*df_uu['Q'], 10**i*df_uu['R'], yerr=10**i*df_uu['dR'], lw=0, elinewidth=1, color=spin_colors['+'])
				s1 = ax.scatter(10*df_uu['Q'], 10**i*df_uu['R'], s=24, color=spin_colors['+'], ec='none', label='R$^{++}$')
				ax.errorbar(10*df_dd['Q'], 10**i*df_dd['R'], yerr=10**i*df_dd['dR'], lw=0, elinewidth=1, color=spin_colors['-'])
				s2 = ax.scatter(10*df_dd['Q'], 10**i*df_dd['R'], s=24, color=spin_colors['-'], ec='none', label='R$^{--}$')

		ax1.legend(frameon=False, ncol=2, loc='upper right', columnspacing=1.4, handletextpad=0.4)
		ax2.legend(frameon=False, ncol=2, loc='upper right', columnspacing=1.4, handletextpad=0.4)
		ax1.set_yscale('log')
		ax2.set_yscale('log')
		format_axis(ax1, 'Q (nm$^{-1}$)', 'Reflectivity', 'R$^{++}$', [q.min(), q.max()])
		format_axis(ax2, 'Q (nm$^{-1}$)', ' ', 'R$^{--}$', [q.min(), q.max()])
		
		fig.tight_layout()
		fig.subplots_adjust(wspace=0.1)
		fig.savefig(image_dir + '/decoded_exp.pdf', bbox_inches='tight')

		ax.legend([s1, p1, s2, p2], ['R$^{++}$', 'Pred.', 'R$^{--}$', 'Pred.'], frameon=False, ncol=2, loc='lower left', columnspacing=1)
		ax.set_yscale('log')
		format_axis(ax, 'Q (nm$^{-1}$)', 'Reflectivity', '', [q.min(), q.max()])
		fig_.tight_layout()
		fig_.savefig(image_dir + '/decoded_exp_' + str(temps[0]) + 'K' + '.pdf', bbox_inches='tight')


		# spin asymmetry
		fig, ax = plt.subplots(1,1, figsize=(5,4.4))
		for i in range(x_pred.shape[0]):
			xdata_uu = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_uu.dat'
			xdata_dd = 'experiments/' + sample_name + '/' + exp_names[i] + '/x_dd.dat'
			df_uu = read_exp(xdata_uu)
			df_dd = read_exp(xdata_dd)
			
			ax.errorbar(10*df_uu['Q'].values, spin_asymmetry([df_uu['R'].values, df_dd['R'].values]),
						yerr=spin_asymmetry_error([df_uu['R'].values, df_dd['R'].values], [df_uu['dR'].values, df_dd['dR'].values]),
						lw=0, elinewidth=1, color=cmap_temp(norm(i)), zorder=len(temps)-i)
			ax.scatter(10*df_uu['Q'].values, spin_asymmetry([df_uu['R'].values, df_dd['R'].values]), s=24,
					   color=cmap_temp(norm(i)), ec='none', label=str(temps[i]) + 'K', zorder=len(temps)-i)

			ax.plot(q, spin_asymmetry([normalize_log_inverse(x_pred[i,:,0], x_moms), normalize_log_inverse(x_pred[i,:,1], x_moms)]),
						color=cmap_temp(norm(i)), zorder=len(temps)-i)

		ax.legend(frameon=False, ncol=2, columnspacing=1)
		ax.locator_params(axis='y', nbins=5)
		format_axis(ax, 'Q (nm$^{-1}$)', 'Spin asymmetry', '', [q.min(), q.max()])
		fig.tight_layout()
		fig.subplots_adjust(wspace=0.1)
		fig.savefig(image_dir + '/decoded_exp_sa.pdf', bbox_inches='tight')

	else:
		fig, ax = plt.subplots(1,1, figsize=(5,4.4))
		fig_, ax_ = plt.subplots(1,1, figsize=(5/1.5,4/2.))

		xdata_uu = 'experiments/' + sample_name + '/' + exp_names[0] + '/x_uu.dat'
		xdata_dd = 'experiments/' + sample_name + '/' + exp_names[0] + '/x_dd.dat'
		df_uu = read_exp(xdata_uu)
		df_dd = read_exp(xdata_dd)
		
		p1, = ax.plot(q, normalize_log_inverse(x_pred[0,:,0], x_moms), color=spin_colors['+'])
		p2, = ax.plot(q, normalize_log_inverse(x_pred[0,:,1], x_moms), color=spin_colors['-'])
		
		ax.errorbar(10*df_uu['Q'], df_uu['R'], yerr=df_uu['dR'], lw=0, elinewidth=1, color=spin_colors['+'])
		s1 = ax.scatter(10*df_uu['Q'], df_uu['R'], s=24, color=spin_colors['+'], ec='none', label='R$^{++}$')
		ax.errorbar(10*df_dd['Q'], df_dd['R'], yerr=df_dd['dR'].values, lw=0, elinewidth=1, color=spin_colors['-'])
		s2 = ax.scatter(10*df_dd['Q'], df_dd['R'], s=24, color=spin_colors['-'], ec='none', label='R$^{--}$')

		ax.legend([s1, p1, s2, p2], ['R$^{++}$', 'Pred.', 'R$^{--}$', 'Pred.'],
				  frameon=False, ncol=2, loc='lower left', columnspacing=1, fontsize=textsize-2)
		ax.set_yscale('log')
		format_axis(ax, 'Q (nm$^{-1}$)', 'Reflectivity', '', [q.min(), q.max()])
		y_major = mpl.ticker.LogLocator(base=10., numticks=10)
		ax.yaxis.set_major_locator(y_major)
		y_minor = mpl.ticker.LogLocator(base=10., subs=np.arange(1.,10.)*0.1, numticks=10)
		ax.yaxis.set_minor_locator(y_minor)
		ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
		ax.tick_params(which='minor', direction='in', length=3)

		fig.tight_layout()
		fig.subplots_adjust(wspace=0.1)
		fig.savefig(image_dir + '/decoded_exp.pdf', bbox_inches='tight')

		# zoom inset
		ax_.plot(q, normalize_log_inverse(x_pred[0,:,0], x_moms), color=spin_colors['+'])
		ax_.plot(q, normalize_log_inverse(x_pred[0,:,1], x_moms), color=spin_colors['-'])

		ax_.errorbar(10*df_uu['Q'], df_uu['R'], yerr=df_uu['dR'], lw=0, elinewidth=1, color=spin_colors['+'])
		s1 = ax_.scatter(10*df_uu['Q'], df_uu['R'], s=24, color=spin_colors['+'], ec='none', label='R$^{++}$')
		ax_.errorbar(10*df_dd['Q'], df_dd['R'], yerr=df_dd['dR'], lw=0, elinewidth=1, color=spin_colors['-'])
		s2 = ax_.scatter(10*df_dd['Q'], df_dd['R'], s=24, color=spin_colors['-'], ec='none', label='R$^{--}$')

		ax_.set_yscale('log')
		format_axis(ax_, 'Q (nm$^{-1}$)', 'Refl.', '', [1, q.max()], [1e-6, 5e-5], nbins=5)
		ax_.tick_params(which='minor', direction='in', length=3)

		fig_.tight_layout()
		fig_.subplots_adjust(wspace=0.1)
		fig_.savefig(image_dir + '/decoded_exp_inset.pdf', bbox_inches='tight')

		# spin asymmetry
		fig, ax = plt.subplots(1,1, figsize=(5,4.4))
		ax.errorbar(10*df_uu['Q'].values, spin_asymmetry([df_uu['R'].values, df_dd['R'].values]),
					yerr=spin_asymmetry_error([df_uu['R'].values, df_dd['R'].values], [df_uu['dR'].values, df_dd['dR'].values]),
					lw=0, elinewidth=1, color=cmap_temp(norm(0)))
		s1 = ax.scatter(10*df_uu['Q'].values, spin_asymmetry([df_uu['R'].values, df_dd['R'].values]), s=24,
				   		color=cmap_temp(norm(0)), ec='none', label='Data', zorder=1)

		p1, = ax.plot(q, spin_asymmetry([normalize_log_inverse(x_pred[0,:,0], x_moms), normalize_log_inverse(x_pred[0,:,1], x_moms)]),
					  color='black', label='Pred.')

		ax.legend([s1, p1], ['Data', 'Pred.'], frameon=False, ncol=2, loc='lower left', columnspacing=1)
		ax.locator_params(axis='y', nbins=5)
		format_axis(ax, 'Q (nm$^{-1}$)', 'Spin asymmetry', '', [q.min(), q.max()])
		fig.tight_layout()
		fig.subplots_adjust(wspace=0.1)
		fig.savefig(image_dir + '/decoded_exp_sa.pdf', bbox_inches='tight')


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


def plot_latent_representation(image_dir, x_data, y_data, y_ids, y_labels, y_units, tag='', z_exp=None, exp_names=None):
	# visualize separability by y-values
	if len(y_ids) > 15: n = 5
	else: n = 3
	w = int((len(y_ids) + (len(y_ids)%n > 0)*(n - len(y_ids)%n))/n)
	fig = plt.figure(figsize=(3*w, n*2.5))
	
	D = np.zeros((len(y_ids), x_data.shape[1]))

	# downsampling
	d = max(1, len(x_data)//5000)
	alpha = 0.3
	
	try: len(z_exp)
	except: pass
	else:
		n_exp = z_exp.shape[0]
		temps = [int(k[:-1]) for k in exp_names]
		if len(temps) > 1: norm = mpl.colors.Normalize(vmin=0, vmax=len(temps)-1)
		else: norm = mpl.colors.Normalize(vmin=0, vmax=1)
	
	xd = x_data[::d,:]
	for k in range(len(y_ids)):
		ax = fig.add_subplot(n, w, k+1)
		yd = y_data[::d,y_ids[k]]
		
		Dy = gradient_pt_cloud(xd, yd)
		dy = np.abs(Dy.mean(axis=0))
		dy = (dy - dy.min())/(dy.max() - dy.min())
		D[k,:] = dy
		idx = np.argsort(dy)[::-1]
		x_proj = np.concatenate([xd[:,[idx[0]]], xd[:,[idx[1]]]], axis=1)
		g = ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, cmap=cmap)
		
		cbar = fig.colorbar(g, ax=ax, aspect=12)
		cbar.ax.set_title(y_units[k], fontsize=fontsize)
		cbar.ax.tick_params(direction='in', length=6, width=1)
		cbar.outline.set_visible(False)
		ax.clear()

		ax.scatter(x_proj[:,0], x_proj[:,1], c=yd, s=10, alpha=alpha, cmap=cmap)
		
		try: len(z_exp)
		except: pass
		else:
			z_proj = np.concatenate([z_exp[:,[idx[0]]], z_exp[:,[idx[1]]]], axis=1)
			for i in range(n_exp):
				ax.scatter(z_proj[[i],0], z_proj[[i],1], color=cmap_temp(norm(i)), ec='black', s=36)
				
		ax.text(0.075, 0.9, y_labels[k], ha='left', va='center', transform=ax.transAxes)

		format_axis(ax, '', '')
		ax.locator_params(tight=True, nbins=4)
		ax.set_xticks([]); ax.set_yticks([])

	fig.tight_layout()
	fig.subplots_adjust(wspace=0.15, hspace=0.2)
	fig.savefig(image_dir + '/' + tag + '.png', dpi=400)

	# save colorbar
	norm = mpl.colors.Normalize(vmin=0, vmax=1)
	sm = mpl.cm.ScalarMappable(cmap=cmap_cm, norm=norm)
	sm.set_array([])

	fig, ax = plt.subplots(figsize=(0.21,0.25*len(y_ids)))
	cbar = fig.colorbar(sm, cax=ax, ticks=[0, 0.5, 1])
	format_axis(cbar.ax, '', '')
	cbar.ax.tick_params(which='major', length=0, width=0)
	cbar.outline.set_visible(False)
	fig.savefig(image_dir + '/' + tag + '_contribution_cbar.pdf', bbox_inches='tight')

	# plot gradient contribution
	fig, ax = plt.subplots(figsize=(0.7*len(y_ids), 0.3*len(y_ids)))
	ax.imshow(D, cmap=cmap_cm)
	format_axis(ax, '', '')
	ax.set_yticks(list(range(len(y_ids)))); ax.set_xticks([])
	ax.set_yticklabels(y_labels)

	plt.xticks(rotation=90)
	fig.tight_layout()
	fig.savefig(image_dir + '/' + tag + '_contribution.pdf', bbox_inches='tight')


def plot_predicted(image_dir, y_pred, y_true, y_ids, y_labels, y_units):
	if len(y_ids) > 15: n = 5
	else: n = 3
	w = int((len(y_ids) + (len(y_ids)%n > 0)*(n - len(y_ids)%n))/n)
	fig = plt.figure(figsize=(3.3*w, n*2.67))
	
	for k in range(len(y_ids)):
		ax = fig.add_subplot(n, w, k+1)
		
		bins = 60
		norm = mpl.colors.LogNorm(vmin=1, vmax=1e2)
		
		_, _, _, g = ax.hist2d(y_true[:,y_ids[k]], y_pred[:,k], bins=bins, cmap=cmap_mse, norm=norm, cmin=1, alpha=0.8, ec='none')
		cbar = fig.colorbar(g, ax=ax, aspect=12)
		cbar.ax.tick_params(direction='in', length=6, width=1)
		cbar.ax.tick_params(which='minor', length=0)
		cbar.outline.set_visible(False)
		
		format_axis(ax, ' ', ' ', title=y_labels[k] + ' ' + y_units[k])
		ax.locator_params(tight=True, nbins=4)
		ax.set_aspect('equal')
		
		x_min = y_true[:,y_ids[k]].min() - 0.05
		x_max = y_true[:,y_ids[k]].max() + 0.05
		ax.set_xlim([x_min, x_max])
		ax.set_ylim([x_min, x_max])
		ax.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--')
		ax.yaxis.labelpad = 0
		
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.4, hspace=0.3)
	fig.savefig(image_dir + '/predicted.png', dpi=400)


def get_fpr_tpr(y_true, y_pred):
    h, x_edges, y_edges = np.histogram2d(y_true, y_pred, bins=60)
    t_min = min(x_edges.min(), y_edges.min())
    t_max = max(x_edges.max(), y_edges.max())
    bins = np.linspace(t_min, t_max, 60)
    
    h, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
    
    th = bins[1:]
    tpr, fpr = [], []
    for i in range(1,len(bins)):
        tpr += [np.zeros(len(range(i,len(bins))))]
        fpr += [np.zeros(len(range(i,len(bins))))]
        for j in range(i,len(bins)):
            tn = h[:j,:i].sum()
            tp = h[j:,i:].sum()
            fn = h[:j,i:].sum()
            fp = h[j:,:i].sum()
            tpr[i-1][j-i] = tp/(tp + fn + 1e-12)
            fpr[i-1][j-i] = fp/(tn + fp + 1e-12)
            
    return fpr, tpr, th
            
            
def get_optimal_threshold(fpr, tpr, th):
    k = np.argmin((fpr - tpr) + np.abs(tpr - (1. - fpr)))
    print('tpr:', tpr[k], 'fpr:', fpr[k], 'tnr:', 1 - fpr[k], 'threshold:', th[k])
    return tpr[k], fpr[k], th[k], k


def get_threshold(image_dir, y_true, y_pred, y_label, y_unit, y_name, show=False):
    fpr, tpr, th = get_fpr_tpr(y_true, y_pred)
    fprs = np.concatenate(fpr)
    tprs = np.concatenate(tpr)
    ths = np.concatenate([th[i:] for i in range(len(th))])
    idx = [i for i in range(len(th)) for j in range(i,len(th))]
    tpr0, fpr0, th0, k = get_optimal_threshold(fprs, tprs, ths)
    th_true = th[idx[k]]
    
    if show:
        norm = mpl.colors.LogNorm(vmin=1, vmax=1e2)
        fig, ax = plt.subplots(1,2, figsize=(10,3.5))
        _, _, _, g = ax[0].hist2d(y_true, y_pred, bins=60, cmap=cmap_mse, norm=norm, alpha=0.8)
        cbar = fig.colorbar(g, ax=ax[0], aspect=12)
        cbar.ax.tick_params(direction='in', length=6, width=1)
        cbar.ax.tick_params(which='minor', length=0)
        cbar.outline.set_visible(False)
        
        ax[0].axvline(th_true, color='black')
        ax[0].axhline(th0, color='red')
        ax[0].locator_params(tight=True, nbins=4)
        ax[0].set_aspect('equal')
        x_min = y_true.min() - 0.05
        x_max = y_true.max() + 0.05
        ax[0].set_xlim([x_min, x_max])
        ax[0].set_ylim([x_min, x_max])
        ax[0].plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--')
        ax[0].yaxis.labelpad = 0
        format_axis(ax[0], 'True', 'Predicted', title=y_label + ' ' + y_unit)
        
        g = ax[1].scatter(fprs, tprs, s=36, c=ths, cmap=cmap)
        ax[1].scatter(fpr0, tpr0, s=48, lw=1.5, ec='black', fc='white', zorder=10)
        cbar = fig.colorbar(g, ax=ax[1], aspect=12)
        cbar.ax.set_ylabel(y_label.split('_')[0] + '_{th}$ ' + y_unit)
        cbar.ax.tick_params(direction='in', length=6, width=1)
        cbar.ax.tick_params(which='minor', length=0)
        cbar.outline.set_visible(False)
        
        ax[1].set_xlim([-0.05,1.05])
        ax[1].set_ylim([-0.05,1.05])
        ax[1].set_aspect('equal')
        format_axis(ax[1], 'False positive rate', 'True positive rate')
        
        fig.subplots_adjust(wspace=0.25)
        fig.savefig(image_dir + '/threshold_' + y_name + '.pdf', bbox_inches='tight')
    return th0


def reject_outliers(x, m=2.):
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return x[s<m]

        
def plot_exp_statistics(image_dir, df_exp, reps, y_name, y_header, y_labels, y_units, y_th=None, y_lims=None):
    # plot statistics of given parameter
    column = y_header.index(y_name)
    df_exp['temperature'] = df_exp['set'].map(lambda x: int(x[:-1]))

    df_exp['p'] = df_exp['y_pred_'].map(lambda x: x[column])
    
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
        format_axis(ax, 'Temperature (K)', y_labels[column] + ' ' + y_units[column], ylims=y_lims)
        fig.subplots_adjust(bottom=0.2)

    else:
        T = np.unique(df_exp['temperature'])[0]
        format_axis(ax, 'T (K)', y_labels[column] + ' ' + y_units[column], ylims=y_lims)
        fig.subplots_adjust(bottom=0.2, left=0.45)

    fig.savefig(image_dir + '/' + y_header[column] + '_distribution_exp.pdf', bbox_inches='tight')
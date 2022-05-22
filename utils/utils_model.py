import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


class TorchStandardScaler:
	def fit(self, x):
		self.mean = x.mean(0, keepdim=True)
		self.std = x.std(0, unbiased=False, keepdim=True)

	def transform(self, x):
		x -= self.mean
		x /= self.std
		return x

	def inverse_transform(self, x):
		x *= self.std
		x += self.mean
		return x


class SE1d(nn.Module):
	def __init__(self, channels, r=4, inspect=False):
		super(SE1d, self).__init__()
		
		# multi-layer perceptron
		self.mlp = nn.Sequential(
			nn.Linear(channels, channels//r, bias=False),
			nn.ReLU(),
			nn.Linear(channels//r, channels, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.mlp(x)
		return x*y


class VAE(nn.Module):
	def __init__(self, height, width, latent_dim, start_filters, kernel_size, pool_size, num_conv, num_dense, slope, drop, beta_1):
		super(VAE, self).__init__()

		# definitions
		self.height = height
		self.width = width
		self.latent_dim = latent_dim
		self.b1 = beta_1
		
		# round up kernel size to nearest odd integer
		kernel_size = kernel_size + 1 - kernel_size%2

		## build encoder
		# convolution and pooling layers
		in_channels = 1
		in_height = height
		self.encoder_conv2d = nn.Sequential(*self.conv2d_pool_block(in_channels, start_filters, kernel_size, pool_size, slope, drop))

		modules = []
		in_channels = start_filters
		in_height = in_height//pool_size
		for i in range(1,num_conv):
			modules.append(
				nn.Sequential(*self.conv1d_pool_block(in_channels, start_filters*(2**i), kernel_size, pool_size, slope, drop))
			)
			in_channels = start_filters*(2**i)
			in_height = in_height//pool_size

		self.encoder_conv1d = nn.Sequential(*modules)
		self.in_channels = in_channels
		self.in_height = in_height 

		# fully-connected layers
		modules = []
		in_features = in_height*in_channels
		hidden_dim = 1<<(latent_dim-1).bit_length()
		r = int(np.log(in_features/hidden_dim)/np.log(num_dense))
		for i in range(num_dense-1,0,-1):
			modules.append(
				nn.Sequential(*self.linear_block(in_features, hidden_dim*(r**i), slope, drop))
			)
			in_features = hidden_dim*(r**i)

		self.encoder_fc = nn.Sequential(*modules)
		
		# latent space layers
		self.z_mean = nn.Linear(in_features, latent_dim)
		self.z_log_var = nn.Linear(in_features, latent_dim)

		## build decoder
		# fully-connected layers
		modules = []
		in_features = latent_dim
		for i in range(1,num_dense):
			modules.append(
				nn.Sequential(*self.linear_block(in_features, hidden_dim*(r**i), slope, drop))
			)
			in_features = hidden_dim*(r**i)

		modules.append(
			nn.Sequential(*self.linear_block(in_features, in_channels*in_height, slope, drop))
		)

		self.decoder_fc = nn.Sequential(*modules)

		# convolution and upsampling layers
		modules = []
		for i in range(num_conv-2,-1,-1):
			modules.append(
				nn.Sequential(*self.conv1d_upsample_block(in_channels, start_filters*(2**i), kernel_size, pool_size, slope, drop))
			)
			in_channels = start_filters*(2**i)

		self.decoder_conv = nn.Sequential(*modules)

		# final output layer
		self.recon = nn.Sequential(
						nn.ConvTranspose1d(in_channels=in_channels, out_channels=width, kernel_size=kernel_size,
										   stride=pool_size, padding=(kernel_size-pool_size+1)//2, output_padding=1),
						nn.Sigmoid()
					)

	def conv2d_pool_block(self, in_channels, out_channels, kernel_size, pool_size, slope, drop):
		nn.Conv2d(1, 1, kernel_size=(1,2), stride=1, padding=0, bias=False)
		modules = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,2),
							 stride=(pool_size,1), padding=((kernel_size-pool_size+1)//2,0), bias=False)]

		modules.append(nn.LeakyReLU(negative_slope=slope))
		if drop: modules.append(nn.Dropout(p=drop))
			
		return modules

	def conv1d_pool_block(self, in_channels, out_channels, kernel_size, pool_size, slope, drop):
		modules = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
							 stride=pool_size, padding=(kernel_size-pool_size+1)//2)]
		
		modules.append(nn.LeakyReLU(negative_slope=slope))
		if drop: modules.append(nn.Dropout(p=drop))
			
		return modules

	def conv1d_upsample_block(self, in_channels, out_channels, kernel_size, pool_size, slope, drop):
		modules = [nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
									  stride=pool_size, padding=(kernel_size-pool_size+1)//2, output_padding=1)]
		
		modules.append(nn.LeakyReLU(negative_slope=slope))
		if drop: modules.append(nn.Dropout(p=drop))

		return modules

	def linear_block(self, in_features, out_features, slope, drop):
		modules = [nn.Linear(in_features=in_features, out_features=out_features)]

		modules.append(nn.LeakyReLU(negative_slope=slope))
		if drop: modules.append(nn.Dropout(p=drop))

		return modules
	
	def encode(self, x):
		x = self.encoder_conv2d(x).squeeze(-1)
		x = self.encoder_conv1d(x)
		x = self.encoder_fc(x.view(-1, self.in_channels*self.in_height))
		return self.z_mean(x), self.z_log_var(x)

	def sampling(self, z_mean, z_log_var):
		z_std = torch.exp(0.5*z_log_var)
		eps = torch.randn_like(z_std)
		return z_mean + eps*z_std

	def decode(self, z):
		x = self.decoder_fc(z)
		x = self.decoder_conv(x.view(-1, self.in_channels, self.in_height))
		return self.recon(x).view(-1, self.width, self.height, 1).permute((0,3,2,1))
	
	def recon_loss(self, x_pred, x_true):
		return F.mse_loss(x_pred, x_true)*self.height*self.width
	
	def kld_loss(self, z_mean, z_log_var, z_true):
		return -0.5*torch.sum(1 + z_log_var - (z_mean - z_true).pow(2) - z_log_var.exp(), dim=-1)

	def loss_function(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
		recon_loss = self.recon_loss(d_pred[0], d_true[0])
		kld_loss = self.kld_loss(z_mean, z_log_var, z_true)
		return torch.mean(recon_loss + self.b1*kld_loss)

	def metrics(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
		total_loss = self.loss_function(d_pred, d_true, z_mean, z_log_var, z_true)
		recon_loss = self.recon_loss(d_pred[0], d_true[0])
		kld_loss = torch.mean(self.kld_loss(z_mean, z_log_var, z_true))
		recon_mse = F.mse_loss(d_pred[0], d_true[0])
		return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'recon_mse': recon_mse}

	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		if self.training:
			z = self.sampling(z_mean, z_log_var)
			return [self.decode(z)], z_mean, z_log_var
		else:
			return [self.decode(z_mean)], z_mean, z_log_var
		

class CVAE(VAE):
	def __init__(self, height, width, num_features, scaler, latent_dim, start_filters, kernel_size, pool_size, 
				 num_conv, num_dense, slope, drop, beta_1, beta_2):
		super(CVAE, self).__init__(height, width, latent_dim, start_filters, kernel_size, pool_size,
								   num_conv, num_dense, slope, drop, beta_1)
		
		# definitions
		self.num_features = num_features
		self.scaler = scaler
		self.b2 = beta_2

		# build regressor
		self.reg = nn.Sequential(
					SE1d(latent_dim),
					nn.Linear(in_features=latent_dim, out_features=num_features)
				)
		
		if self.scaler != None:
			self.pos = nn.ReLU()

	def label_loss(self, y_pred, y_true):
		return torch.sum(torch.tensor(self.b2).to(y_pred.device)*F.mse_loss(y_pred, y_true, reduction='none'), dim=-1)

	def loss_function(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
		recon_loss = self.recon_loss(d_pred[0], d_true[0])
		kld_loss = self.kld_loss(z_mean, z_log_var, z_true)
		label_loss = self.label_loss(d_pred[1], d_true[1])
		return torch.mean(recon_loss + self.b1*kld_loss + label_loss)
	
	def metrics(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
		total_loss = self.loss_function(d_pred, d_true, z_mean, z_log_var, z_true)
		recon_loss = self.recon_loss(d_pred[0], d_true[0])
		kld_loss = torch.mean(self.kld_loss(z_mean, z_log_var, z_true))
		label_loss = torch.mean(self.label_loss(d_pred[1], d_true[1]))
		recon_mse = F.mse_loss(d_pred[0], d_true[0])
		label_mse = F.mse_loss(d_pred[1], d_true[1])
		return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'label_loss': label_loss,
				'recon_mse': recon_mse, 'label_mse': label_mse}
	
	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		y = self.reg(z_mean)

		if self.scaler != None:
			y = self.scaler.transform(self.pos(self.scaler.inverse_transform(y)))
		
		if self.training:
			z = self.sampling(z_mean, z_log_var)
			return [self.decode(z), y], z_mean, z_log_var
		else:
			return [self.decode(z_mean), y], z_mean, z_log_var


def init_model(model_name, height, width, num_features, kwargs, device, scaler=None):
	if model_name == 'cvae':
		model = CVAE(height, width, num_features, scaler, **kwargs).to(device)
		metric_keys = ['total_loss', 'recon_loss', 'kld_loss', 'label_loss', 'recon_mse', 'label_mse']
	
	else:
		if 'beta_2' in list(kwargs.keys()): kwargs.pop('beta_2')
		model = VAE(height, width, **kwargs).to(device)
		metric_keys = ['total_loss', 'recon_loss', 'kld_loss', 'recon_mse']
	
	opt = optim.Adam(model.parameters(), lr=1e-3)
	return model, opt, metric_keys


def train(epoch, model, opt, metric_keys, dataloader, device, model_name, y_ids, z_std_norm=True):
	model.train()

	for i, (x_true, y_true) in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
		x_true = x_true.to(device)
		y_true = y_true[:,y_ids].to(device)
		d_pred, z_mean, z_log_var = model(x_true)

		if z_std_norm: z_true = 0
		else:
			z_true = torch.zeros((len(x_true), model.latent_dim)).to(device)
			z_true[:,:y_true.size()[1]] = y_true
		loss = model.loss_function(d_pred, [x_true, y_true], z_mean, z_log_var, z_true)

		# advance
		opt.zero_grad()
		loss.backward()
		opt.step()


def evaluate(epoch, model, metric_keys, dataloader, device, model_name, y_ids, z_std_norm=True):
	model.eval()

	metrics = dict.fromkeys(metric_keys, 0)
	with torch.no_grad():
		for i, (x_true, y_true) in enumerate(dataloader):
			x_true = x_true.to(device)
			y_true = y_true[:,y_ids].to(device)
			d_pred, z_mean, z_log_var = model(x_true)

			if z_std_norm: z_true = 0
			else:
				z_true = torch.zeros((len(x_true), model.latent_dim)).to(device)
				z_true[:,:y_true.size()[1]] = y_true
			metrics_batch = model.metrics(d_pred, [x_true, y_true], z_mean, z_log_var, z_true)
		
			# update metrics
			for key in metrics:
				metrics[key] += metrics_batch[key].cpu().item()

	# print metrics
	disp_keys, disp_values = [], []
	for key in metrics:
		metrics[key] /= len(dataloader)
		disp_keys.append(key + ': {:.2e}')
		disp_values.append(metrics[key])

	print('  '.join(disp_keys).format(*disp_values), flush=True)
	return metrics


def predict(model, dataloader, device, y_ids, scaler, x_set, z_set, dz_set, x_mse, y_set=None, y_err=None):
	model.eval()

	with torch.no_grad():
		i_start = 0
		for i, (x_true, y_true) in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
			x_true = x_true.to(device)
			y_true = y_true[:,y_ids].to(device)
			d_pred, z_mean, z_log_var = model(x_true)
				
			# save to array
			x_set[i_start:i_start + len(x_true),:] = d_pred[0].cpu().squeeze().numpy()
			z_set[i_start:i_start + len(x_true),:] = z_mean.cpu().numpy()
			dz_set[i_start:i_start + len(x_true),:] = np.sqrt(np.exp(z_log_var.cpu().numpy()))
			x_mse[i_start:i_start + len(x_true)] = F.mse_loss(d_pred[0].squeeze(), x_true.squeeze(),
															  reduction='none').mean(dim=(1,2)).cpu().numpy()

			# save y_pred
			try: len(y_set)
			except: pass
			else:
				if scaler: y_set[i_start:i_start + len(x_true),:] = scaler.inverse_transform(d_pred[1]).cpu().numpy()
				else: y_set[i_start:i_start + len(x_true),:] = d_pred[1].cpu().numpy()
			
			# save y_err
			try: len(y_err)
			except: pass
			else:
				metric = F.l1_loss(d_pred[1], y_true, reduction='none')
				if scaler: y_err[i_start:i_start + len(x_true),:] = (scaler.inverse_transform(metric) - scaler.mean).cpu().numpy()
				else: y_err[i_start:i_start + len(x_true),:] = metric.cpu().numpy()

			i_start += len(x_true)


def predict_exp(model, x_true, device, scaler, x_set, z_set, dz_set, x_mse, y_set=None):
	model.eval()

	with torch.no_grad():
		x_true = x_true.to(device)
		d_pred, z_mean, z_log_var = model(x_true)

		# save to array
		x_set[:,:] = d_pred[0].cpu().squeeze().numpy()
		z_set[:,:] = z_mean.cpu().numpy()
		dz_set[:,:] = np.sqrt(np.exp(z_log_var.cpu().numpy()))
		x_mse[:] = F.mse_loss(d_pred[0].squeeze(1), x_true.squeeze(1), reduction='none').mean(dim=(1,2)).cpu().numpy()

		# save y_pred
		try: len(y_set)
		except: pass
		else:
			if scaler: y_set[:,:] = scaler.inverse_transform(d_pred[1]).cpu().numpy()
			else: y_set[:,:] = d_pred[1].cpu().numpy()


def get_predictions(model, dataloaders, d_sets, device, height, width, num_features, kwargs, model_name, y_ids, scaler):
	
	df = pd.DataFrame({'set': [j for i in [[d_set]*len(dataloaders[d_set].dataset) for d_set in d_sets] for j in i]})

	xs, x_preds, x_mses, zs, dzs, ys, y_preds, y_errs = [], [], [], [], [], [], [], []
	for d_set in d_sets:
		# initialize arrays
		x_pred = np.zeros((len(dataloaders[d_set].dataset), height, width))
		x_mse = np.zeros((len(dataloaders[d_set].dataset),))
		z = np.zeros((len(dataloaders[d_set].dataset), kwargs['latent_dim']))
		dz = np.zeros((len(dataloaders[d_set].dataset), kwargs['latent_dim']))

		if model_name == 'cvae':
			y_pred = np.zeros((len(dataloaders[d_set].dataset), num_features))
			y_err = np.zeros((len(dataloaders[d_set].dataset), num_features))

			# predict
			predict(model, dataloaders[d_set], device, y_ids, scaler, x_pred, z, dz, x_mse, y_pred, y_err=y_err)
		else:
			y_pred = None
			y_err = None

			# predict
			predict(model, dataloaders[d_set], device, y_ids, scaler, x_pred, z, dz, x_mse)
		
		x = torch.clone(dataloaders[d_set].dataset[:][0]).numpy()
		y = torch.clone(dataloaders[d_set].dataset[:][1]).to(device)
		y[:,y_ids] = scaler.inverse_transform(y[:,y_ids])
		y = y.cpu().numpy()
		
		xs += [x]
		ys += [y]
		x_preds += [x_pred]
		zs += [z]
		dzs += [dz]
		x_mses += [x_mse]
		y_preds += [y_pred]
		y_errs += [y_err]
	
	df['x_true'] = [k for k in np.concatenate(xs, axis=0).squeeze()]
	df['x_pred'] = [k for k in np.concatenate(x_preds, axis=0).squeeze()]
	df['x_mse'] = [k for k in np.concatenate(x_mses, axis=0)]
	df['z'] = [k for k in np.concatenate(zs, axis=0)]
	df['dz'] = [k for k in np.concatenate(dzs, axis=0)]
	df['y_true'] = [k for k in np.concatenate(ys, axis=0)]

	try: len(y_preds[0])
	except: pass
	else: df['y_pred'] = [k for k in np.concatenate(y_preds, axis=0)]

	try: len(y_errs[0])
	except: pass
	else: df['y_err'] = [k for k in np.concatenate(y_errs, axis=0)]
	
	return df


def get_predictions_exp(model, x_exp, dx_exp, exp_names, device, height, width, num_features, kwargs, model_name, y_ids, scaler):
	df = pd.DataFrame({'set': exp_names, 'x_true': [k for k in x_exp.squeeze(1).numpy()], 'dx': [k for k in dx_exp]})
	
	x_exp_pred = np.zeros((x_exp.shape[0], height, width))
	z_exp = np.zeros((x_exp.shape[0], kwargs['latent_dim']))
	dz_exp = np.zeros((x_exp.shape[0], kwargs['latent_dim']))
	x_exp_mse = np.zeros((x_exp.shape[0],))
	
	if model_name == 'cvae':
		y_exp_pred = np.zeros((x_exp.shape[0], num_features))
		predict_exp(model, x_exp, device, scaler, x_exp_pred, z_exp, dz_exp, x_exp_mse, y_exp_pred)

	else:
		y_exp_pred = None
		predict_exp(model, x_exp, device, scaler, x_exp_pred, z_exp, dz_exp, x_exp_mse)
	
	df['x_pred'] = [k for k in x_exp_pred]
	df['x_mse'] = [k for k in x_exp_mse]
	df['z'] = [k for k in z_exp]
	df['dz'] = [k for k in dz_exp]

	try: len(y_exp_pred)
	except: pass
	else: df['y_pred'] = [k for k in y_exp_pred]
	
	return df
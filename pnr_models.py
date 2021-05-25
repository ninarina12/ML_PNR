import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class VAE(nn.Module):
    def __init__(self, height, width, hidden_dim, latent_dim, start_filters, kernel_size, pool_size, num_conv,
                 num_dense, slope, drop, beta_1):
        super(VAE, self).__init__()

        # definitions
        self.height = height
        self.width = width
        self.b1 = beta_1
        
        # round up kernel size to nearest odd integer
        kernel_size = kernel_size + 1 - kernel_size%2

        ## build encoder
        # convolution and pooling layers
        modules = []
        in_channels = 1
        in_height = height
        for i in range(num_conv):
            modules.append(
                nn.Sequential(*self.conv2d_pool_block(in_channels, start_filters*(2**i), kernel_size, pool_size,
                                                      slope, drop))
            )
            in_channels = start_filters*(2**i)
            in_height = in_height//pool_size

        self.encoder_conv = nn.Sequential(*modules)
        self.in_channels = in_channels
        self.in_height = in_height 
        self.in_width = width

        # fully-connected layers
        modules = []
        in_features = in_height*width*in_channels
        for i in range(num_dense):
            modules.append(
                nn.Sequential(*self.linear_block(in_features, hidden_dim//(2**i), slope, drop))
            )
            in_features = hidden_dim//(2**i)

        self.encoder_fc = nn.Sequential(*modules)
        
        # latent space layers
        self.z_mean = nn.Linear(in_features, latent_dim)
        self.z_log_var = nn.Linear(in_features, latent_dim)

        ## build decoder
        # fully-connected layers
        modules = []
        in_features = latent_dim
        for i in range(num_dense-1,-1,-1):
            modules.append(
                nn.Sequential(*self.linear_block(in_features, hidden_dim//(2**i), slope, drop))
            )
            in_features = hidden_dim//(2**i)

        modules.append(
            nn.Sequential(*self.linear_block(in_features, in_channels*in_height*width, slope, drop))
        )

        self.decoder_fc = nn.Sequential(*modules)

        # convolution and upsampling layers
        modules = []
        for i in range(num_conv-2,-1,-1):
            modules.append(
                nn.Sequential(*self.conv2d_upsample_block(in_channels, start_filters*(2**i), kernel_size, pool_size,
                                                          slope, drop))
            )
            in_channels = start_filters*(2**i)

        self.decoder_conv = nn.Sequential(*modules)

        # final output layer
        self.recon = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=1, kernel_size=(kernel_size,1),
                                           stride=(pool_size,1), padding=((kernel_size-pool_size+1)//2,0),
                                           output_padding=(1,0)),
                        nn.Sigmoid()
                    )

    def conv2d_pool_block(self, in_channels, out_channels, kernel_size, pool_size, slope, drop):
        modules = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,1),
                             stride=(pool_size,1), padding=((kernel_size-pool_size+1)//2, 0))]
        
        modules.append(nn.LeakyReLU(negative_slope=slope))
        if drop: modules.append(nn.Dropout(p=drop))
            
        return modules

    def conv2d_upsample_block(self, in_channels, out_channels, kernel_size, pool_size, slope, drop):
        modules = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,1),
                                      stride=(pool_size,1), padding=((kernel_size-pool_size+1)//2,0),
                                      output_padding=(1,0))]
        
        modules.append(nn.LeakyReLU(negative_slope=slope))
        if drop: modules.append(nn.Dropout(p=drop))

        return modules

    def linear_block(self, in_features, out_features, slope, drop):
        modules = [nn.Linear(in_features=in_features, out_features=out_features)]

        modules.append(nn.LeakyReLU(negative_slope=slope))
        if drop: modules.append(nn.Dropout(p=drop))

        return modules
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x.view(-1, self.in_channels*self.in_height*self.in_width))
        return self.z_mean(x), self.z_log_var(x)

    def sampling(self, z_mean, z_log_var):
        z_std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(z_std)
        return z_mean + eps*z_std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = self.decoder_conv(x.view(-1, self.in_channels, self.in_height, self.in_width))
        return self.recon(x)
    
    def recon_loss(self, x_pred, x_true):
        return F.mse_loss(x_pred, x_true)*self.height*self.width
        
    def kld_loss(self, z_mean, z_log_var, z_true):
        return -0.5*torch.sum(1 + z_log_var - (z_mean - z_true).pow(2) - z_log_var.exp(), dim=-1)
    
    def loss_function(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        recon_loss = self.recon_loss(d_pred[0], d_true[0])
        kld_loss = self.kld_loss(z_mean, z_log_var, z_true)
        return torch.mean(recon_loss + self.b1*kld_loss)

    def metrics(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        total_loss = self.loss_function(d_pred[0], d_true[0], z_mean, z_log_var, z_true)
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
    def __init__(self, height, width, hidden_dim, latent_dim, start_filters, kernel_size, pool_size, 
                 num_conv, num_dense, slope, drop, beta_1, beta_2):
        super(CVAE, self).__init__(height, width, hidden_dim, latent_dim, start_filters, kernel_size, pool_size,
                                   num_conv, num_dense, slope, drop, beta_1)
        
        # definitions
        self.b2 = beta_2
        
        # build classifier
        self.classifier = nn.Sequential(
                            nn.Linear(in_features=latent_dim, out_features=1, bias=False),
                            nn.Sigmoid())
    
    def label_loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)    
    
    def class_loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
    
    def loss_function(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        recon_loss = self.recon_loss(d_pred[0], d_true[0])
        kld_loss = self.kld_loss(z_mean, z_log_var, z_true)
        try: len(z_true)
        except: label_loss = 0
        else: label_loss = self.label_loss(z_mean, z_true)
        class_loss = self.class_loss(d_pred[1], d_true[1])
        return torch.mean(recon_loss + self.b1*kld_loss) + self.b2*(label_loss + class_loss)
    
    def metrics(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        total_loss = self.loss_function(d_pred[0], d_true[0], z_mean, z_log_var, z_true)
        recon_loss = self.recon_loss(d_pred[0], d_true[0])
        kld_loss = torch.mean(self.kld_loss(z_mean, z_log_var, z_true))
        try: len(z_true)
        except: label_loss = 0
        else: label_loss = self.label_loss(z_mean, z_true)
        class_loss = self.class_loss(d_pred[1], d_true[1])
        recon_mse = F.mse_loss(d_pred[0], d_true[0])
        label_mse = F.mse_loss(z_mean, z_true)
        class_accuracy = ((d_pred[1] > 0.5).float() == d_true[1]).float().mean()
        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'label_loss': label_loss,
                'class_loss': class_loss, 'recon_mse': recon_mse, 'label_mse': label_mse,
                'class_accuracy': class_accuracy}
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        y = self.classifier(z_mean)
        if self.training:
            z = self.sampling(z_mean, z_log_var)
            return [self.decode(z), y], z_mean, z_log_var
        else:
            return [self.decode(z_mean), y], z_mean, z_log_var
        

class RVAE(CVAE):
    def __init__(self, height, width, num_features, hidden_dim, latent_dim, start_filters, kernel_size, pool_size, 
                 num_conv, num_dense, slope, drop, beta_1, beta_2):
        super(RVAE, self).__init__(height, width, hidden_dim, latent_dim, start_filters, kernel_size, pool_size,
                                   num_conv, num_dense, slope, drop, beta_1, beta_2)
        
        # definitions
        self.num_features = num_features
        
        # build regressors
        for i in range(num_features):
            p = nn.Linear(in_features=latent_dim, out_features=1, bias=False)
            setattr(self, "reg_%d"%i, p)
    
    def loss_function(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        recon_loss = self.recon_loss(d_pred[0], d_true[0])
        kld_loss = self.kld_loss(z_mean, z_log_var, z_true)
        label_loss = self.label_loss(d_pred[1][:,:-1], d_true[1][:,:-1])
        class_loss = self.class_loss(d_pred[1][:,[-1]], d_true[1][:,[-1]])
        return torch.mean(recon_loss + self.b1*kld_loss) + self.b2*(label_loss + class_loss)
    
    def metrics(self, d_pred, d_true, z_mean, z_log_var, z_true=0):
        total_loss = self.loss_function(d_pred[0], d_true[0], z_mean, z_log_var, z_true)
        recon_loss = self.recon_loss(d_pred[0], d_true[0])
        kld_loss = torch.mean(self.kld_loss(z_mean, z_log_var, z_true))
        label_loss = self.label_loss(d_pred[1][:,:-1], d_true[1][:,:-1])
        class_loss = self.class_loss(d_pred[1][:,[-1]], d_true[1][:,[-1]])
        recon_mse = F.mse_loss(d_pred[0], d_true[0])
        label_mse = F.mse_loss(d_pred[1], d_true[1])
        class_accuracy = ((d_pred[1][:,[-1]] > 0.5).float() == d_true[1][:,[-1]]).float().mean()
        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'label_loss': label_loss,
                'class_loss': class_loss, 'recon_mse': recon_mse, 'label_mse': label_mse,
                'class_accuracy': class_accuracy}
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        y = self.classifier(z_mean)
        y = torch.cat([getattr(self, "reg_%d"%i)(z_mean) for i in range(self.num_features)] + [y], dim=1)
        if self.training:
            z = self.sampling(z_mean, z_log_var)
            return [self.decode(z), y], z_mean, z_log_var
        else:
            return [self.decode(z_mean), y], z_mean, z_log_var

#################################################################################################################

def init_model(model_name, height, width, num_features, kwargs, device):
    if model_name == 'rvae':
        model = RVAE(height, width, num_features, **kwargs).to(device)
        metric_keys = ['total_loss', 'recon_loss', 'kld_loss', 'label_loss', 'class_loss', 'recon_mse', 'label_mse',
                       'class_accuracy']
    
    elif model_name == 'cvae':
        model = CVAE(height, width, **kwargs).to(device)
        metric_keys = ['total_loss', 'recon_loss', 'kld_loss', 'label_loss', 'class_loss', 'recon_mse', 'label_mse',
                       'class_accuracy']
        
    else:
        kwargs.pop('beta_2')
        model = VAE(height, width, **kwargs).to(device)
        metric_keys = ['total_loss', 'recon_loss', 'kld_loss', 'recon_mse']
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    return model, opt, metric_keys


def train(epoch, model, opt, metric_keys, train_loader, device, model_name, z_std_norm, y_ids):
    model.train()

    metrics = dict.fromkeys(metric_keys, 0)
    for i, (x_true, y_true) in enumerate(train_loader):
        x_true = x_true.to(device)
        y_true = y_true[:,y_ids].to(device)
        d_pred, z_mean, z_log_var = model(x_true)
        
        if z_std_norm: z_true = 0
        else: z_true = y_true[:,:-1]
            
        if model_name == 'cvae':
            loss = model.loss_function(d_pred, [x_true, y_true[:,[-1]]], z_mean, z_log_var, z_true)
            metrics_batch = model.metrics(d_pred, [x_true, y_true[:,[-1]]], z_mean, z_log_var, z_true)
        else:
            loss = model.loss_function(d_pred, [x_true, y_true], z_mean, z_log_var, z_true)
            metrics_batch = model.metrics(d_pred, [x_true, y_true], z_mean, z_log_var, z_true)
        
        # update metrics
        for key in metrics:
            metrics[key] += metrics_batch[key].detach().cpu().item()

        # advance
        opt.zero_grad()
        loss.backward()
        opt.step()

    # print metrics
    disp_keys = ['epoch: {} train =====']
    disp_values = [epoch]
    for key in metrics:
        metrics[key] /= len(train_loader)
        disp_keys.append(key + ': {:.2e}')
        disp_values.append(metrics[key])

    print('   '.join(disp_keys).format(*disp_values))
    return metrics


def evaluate(epoch, model, metric_keys, valid_loader, device, model_name, z_std_norm, y_ids):
    model.eval()

    metrics = dict.fromkeys(metric_keys, 0)
    with torch.no_grad():
        for i, (x_true, y_true) in enumerate(valid_loader):
            x_true = x_true.to(device)
            y_true = y_true[:,y_ids].to(device)
            d_pred, z_mean, z_log_var = model(x_true)
            
            if z_std_norm: z_true = 0
            else: z_true = y_true[:,:-1]
            
            if model_name == 'cvae':
                metrics_batch = model.metrics(d_pred, [x_true, y_true[:,[-1]]], z_mean, z_log_var, z_true)
            else:
                metrics_batch = model.metrics(d_pred, [x_true, y_true], z_mean, z_log_var, z_true)
        
            # update metrics
            for key in metrics:
                metrics[key] += metrics_batch[key].cpu().item()

    # print metrics
    disp_keys = ['epoch: {} valid =====']
    disp_values = [epoch]
    for key in metrics:
        metrics[key] /= len(valid_loader)
        disp_keys.append(key + ': {:.2e}')
        disp_values.append(metrics[key])

    print('   '.join(disp_keys).format(*disp_values), flush=True)
    return metrics


def predict(model, data_loader, device, y_ids, x_set, z_set, x_mse, y_set=None, y_mse=None, z_mse=None):
    model.eval()

    with torch.no_grad():
        i_start = 0
        for i, (x_true, y_true) in enumerate(data_loader):
            x_true = x_true.to(device)
            y_true = y_true[:,y_ids].to(device)
            d_pred, z_mean, z_log_var = model(x_true)
                
            # save to array
            x_set[i_start:i_start + len(x_true),:] = d_pred[0].cpu().squeeze().numpy()
            z_set[i_start:i_start + len(x_true),:] = z_mean.cpu().numpy()
            x_mse[i_start:i_start + len(x_true)] = F.mse_loss(d_pred[0].squeeze(), x_true.squeeze(),
                                                              reduction='none').mean(dim=(1,2)).cpu().numpy()

            # save y_pred
            try: len(y_set)
            except: pass
            else: y_set[i_start:i_start + len(x_true),:] = d_pred[1].cpu().numpy()
            
            # save y_mse
            try: len(y_mse)
            except: pass
            else:
                metric = F.mse_loss(d_pred[1][:,:-1], y_true[:,:-1], reduction='none')
                y_mse[i_start:i_start + len(x_true),:] = metric.cpu().numpy()

            # save z_mse
            try: len(z_mse)
            except: pass
            else:
                metric = F.mse_loss(z_mean, y_true[:,:-1], reduction='none')
                z_mse[i_start:i_start + len(x_true),:] = metric.cpu().numpy()
                
            i_start += len(x_true)


def predict_exp(model, x_true, device, x_set, z_set, x_mse, y_set=None):
    model.eval()

    with torch.no_grad():
        x_true = x_true.to(device)
        d_pred, z_mean, z_log_var = model(x_true)

        # save to array
        x_set[:,:] = d_pred[0].cpu().squeeze().numpy()
        z_set[:,:] = z_mean.cpu().numpy()
        x_mse[:] = F.mse_loss(d_pred[0].squeeze(), x_true.squeeze(), reduction='none').mean(dim=(1,2)).cpu().numpy()

        # save y_pred
        try: len(y_set)
        except: pass
        else: y_set[:,:] = d_pred[1].cpu().numpy()

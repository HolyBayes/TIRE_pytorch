import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from tqdm import trange
import numpy as np

from TIRE import utils

class AbstractAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 window_size: int = 20,
                 latent_dim: int = 1,
                 nr_ae: int = 3,
                 nr_shared: int = 1,
                 loss_weight: float = 1
                 ):
        super().__init__()
        """
        Abstract PyTorch model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.

        Args:
            input_dim: single tick dimension
            window_size: window size for the AE
            latent_dim: latent dimension of AE
            nr_ae: number of parallel AEs (K in paper)
            nr_shared: number of shared features (should be <= latent_dim)
            loss_weight: lambda in paper

        Returns:
            A parallel AE model instance, its encoder part and its decoder part
        """
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.nr_shared = nr_shared
        self.nr_ae = nr_ae
        self.loss_weight = loss_weight


    def encode(self, x):
        raise NotImplemented

    def decode(self, z):
        raise NotImplemented

    def loss(self, x):
        z_shared, z_unshared, z = self.encode(x)
        x_decoded = self.decode(z)
        batch_size = x.shape[0]
        mse_loss = F.mse_loss(x.view(batch_size, self.nr_ae, self.window_size, self.input_dim), x_decoded)

        shared_loss = F.mse_loss(z_shared[:, 1:, :], z_shared[:, :self.nr_ae - 1, :])

        return mse_loss + self.loss_weight * shared_loss

    @property
    def device(self):
        return next(self.parameters()).device

    def fit(self, windows, epoches=200, shuffle=True, batch_size=64, lr=1e-3):
        device = self.device
        windows = self.prepare_input_windows(windows)
        dataset = TensorDataset(torch.from_numpy(windows).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        opt = optim.AdamW(self.parameters(), lr=lr)
        tbar = trange(epoches, desc='Loss: ', leave=True)

        for epoch in tbar:
            for batch_X in dataloader:
                batch_X = batch_X[0].to(device).float()
                opt.zero_grad()
                loss = self.loss(batch_X)
                loss.backward()
                opt.step()
                tbar.set_description(f"Loss: {loss.item() :.2f}")
                tbar.refresh()  # to show immediately the update

    def encode_windows(self, windows):
        device = self.device
        new_windows = torch.from_numpy(self.prepare_input_windows(windows)).float().to(device)
        with torch.no_grad():
            _, _, encoded_windows_pae = self.encode(new_windows)
        encoded_windows_pae = encoded_windows_pae.detach().cpu().numpy()
        encoded_windows = np.concatenate((encoded_windows_pae[:, 0, :self.nr_shared],
                                          encoded_windows_pae[-self.nr_ae + 1:, self.nr_ae - 1, :self.nr_shared]),
                                         axis=0)

        return encoded_windows

    def prepare_input_windows(self, windows):
        """
        Prepares input for create_parallel_ae
        Args:
            windows: list of windows
            nr_ae: number of parallel AEs (K in paper)
        Returns:
            array with shape (nr_ae, (nr. of windows)-K+1, window size)
        """
        new_windows = []
        nr_windows = windows.shape[0]
        for i in range(self.nr_ae):
            new_windows.append(windows[i:nr_windows-self.nr_ae+1+i])
        return np.transpose(new_windows,(1,0,2))


class DenseAE(AbstractAE):
    def __init__(self,
                 input_dim:int,
                 window_size:int=20,
                intermediate_dim:int=0,
                latent_dim:int=1,
                nr_ae:int=3,
                nr_shared:int=1,
                loss_weight:float=1
                ):
        """
        Create a PyTorch model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.

        Args:
            input_dim: single tick dimension
            window_size: window size for the AE
            intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
            latent_dim: latent dimension of AE
            nr_ae: number of parallel AEs (K in paper)
            nr_shared: number of shared features (should be <= latent_dim)
            loss_weight: lambda in paper

        Returns:
            A parallel AE model instance, its encoder part and its decoder part
        """
        super().__init__(input_dim, window_size, latent_dim, nr_ae,nr_shared, loss_weight)
        self.intermediate_dim = intermediate_dim

        if intermediate_dim == 0:
            self.encoder = nn.Identity()
            self.encoder_shared = nn.Sequential(
                nn.Linear(self.input_dim*self.window_size, nr_shared),
                nn.Tanh()
            )
            self.encoder_unshared = nn.Sequential(
                nn.Linear(self.input_dim*self.window_size, latent_dim-nr_shared),
                nn.Tanh()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, self.input_dim*self.window_size),
                nn.Tanh()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim*self.window_size, intermediate_dim),
                nn.ReLU()
            )
            self.encoder_shared = nn.Sequential(
                nn.Linear(intermediate_dim, nr_shared),
                nn.Tanh()
            )
            self.encoder_unshared = nn.Sequential(
                nn.Linear(intermediate_dim, latent_dim-nr_shared),
                nn.Tanh()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, self.input_dim*self.window_size),
                nn.Tanh()
            )
        self.apply(utils.weights_init)
            
    def encode(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.nr_ae, -1)
        z = self.encoder(x)
        z_shared = self.encoder_shared(z)
        z_unshared = self.encoder_unshared(z)
        z = torch.cat([z_shared, z_unshared], -1)
        return z_shared, z_unshared, z
    
    def decode(self, z):
        batch_size = z.shape[0]
        z = self.decoder(z)
        z = z.view(batch_size, self.nr_ae, self.window_size, self.input_dim)
        return z


class AbstractTIRE(nn.Module):
    def __init__(self, input_dim: int, window_size: int = 20,
                 nfft: int = 30,
                 norm_mode: str = 'timeseries',
                 domain: str = 'both',
                 **kwargs):
        """
        Abstract TIRE module

        Args:
            window_size: window size for the AE
            nfft: number of points for DFT
            norm_mode: for calculation of DFT, should the timeseries have mean zero or each window?
            domain: choose from: TD (time domain), FD (frequency domain) or both
            kwargs: AE parameters
        Returns:
            TIRE model consisted of two autoencoders
        """
        super().__init__()
        self.input_dim = input_dim
        self.window_size_td = window_size
        self.AE_TD = AbstractAE(input_dim, self.window_size_td, **kwargs)
        self.window_size_fd = utils.calc_fft(np.random.randn(100, window_size), nfft, norm_mode).shape[-1]
        self.AE_FD = AbstractAE(input_dim, self.window_size_fd, **kwargs)

        self.nfft = nfft
        self.norm_mode = norm_mode
        self.domain = domain

    def fit(self, ts, fit_TD=True, fit_FD=True, **kwargs):
        windows_TD = self.ts_to_windows(ts, self.window_size_td)
        if fit_TD:
            print("Training autoencoder for original timeseries")
            self.AE_TD.fit(windows_TD, **kwargs)

        windows_FD = utils.calc_fft(windows_TD, self.nfft, self.norm_mode)

        if fit_FD:
            print('Training autoencoder for FFT timeseries')
            self.AE_FD.fit(windows_FD, **kwargs)

    def predict(self, ts):
        windows_TD = self.ts_to_windows(ts, self.window_size_td)
        windows_FD = utils.calc_fft(windows_TD, self.nfft, self.norm_mode)
        shared_features_TD, shared_features_FD = [ae.encode_windows(windows) for ae, windows in
                                                  [(self.AE_TD, windows_TD), (self.AE_FD, windows_FD)]]

        dissimilarities = utils.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD,
                                                            self.domain, self.window_size_td)
        scores = utils.change_point_score(dissimilarities, self.window_size_td)
        return dissimilarities, scores

    @staticmethod
    def ts_to_windows(ts, window_size):
        indices = np.arange(ts.shape[0])
        shape = indices.shape[:-1] + (indices.shape[-1] - window_size + 1, window_size)
        strides = indices.strides + (indices.strides[-1],)
        indices = np.lib.stride_tricks.as_strided(indices, shape=shape, strides=strides)
        windows = ts[indices]
        if windows.shape[-1] == 1:
            windows = windows.squeeze(-1)
        return windows

class DenseTIRE(AbstractTIRE):
    def __init__(self, input_dim:int, window_size:int=20,
                 intermediate_dim_TD:int=0, 
                 intermediate_dim_FD:int=10,
                 nfft:int=30,
                 norm_mode:str='timeseries',
                 domain:str='both',
                 **kwargs):
        """
        Create a TIRE model with dense Autoencoders.

        Args:
            window_size: window size for the AE
            intermediate_dim_TD: intermediate dimension for original timeseries AE, for single-layer AE use 0
            intermediate_dim_FD: intermediate dimension for DFT timeseries AE, for single-layer AE use 0
            nfft: number of points for DFT 
            norm_mode: for calculation of DFT, should the timeseries have mean zero or each window?
            domain: choose from: TD (time domain), FD (frequency domain) or both
            kwargs: AE parameters
        Returns:
            TIRE model consisted of two autoencoders
        """
        super().__init__(input_dim, window_size, nfft, norm_mode, domain)
        self.AE_TD = DenseAE(input_dim, self.window_size_td, intermediate_dim=intermediate_dim_TD, **kwargs)
        self.AE_FD = DenseAE(input_dim, self.window_size_fd, intermediate_dim=intermediate_dim_FD, **kwargs)
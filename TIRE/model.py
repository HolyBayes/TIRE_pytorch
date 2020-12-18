import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from tqdm import trange
import numpy as np

from TIRE import utils

class AE(nn.Module):
    def __init__(self, window_size:int=20,
                       intermediate_dim:int=0,
                       latent_dim:int=1,
                       nr_ae:int=3,
                       nr_shared:int=1,
                       loss_weight:float=1
                ):
        super().__init__()        
        """
        Create a PyTorch model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.

        Args:
            window_size: window size for the AE
            intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
            latent_dim: latent dimension of AE
            nr_ae: number of parallel AEs (K in paper)
            nr_shared: number of shared features (should be <= latent_dim)
            loss_weight: lambda in paper

        Returns:
            A parallel AE model instance, its encoder part and its decoder part
        """
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.nr_shared = nr_shared
        self.nr_ae = nr_ae
        self.intermediate_dim = intermediate_dim
        self.loss_weight = loss_weight

        if intermediate_dim == 0:
            self.encoder = nn.Identity()
            self.encoder_shared = nn.Sequential(
                nn.Linear(self.window_size, nr_shared),
                nn.Tanh()
            )
            self.encoder_unshared = nn.Sequential(
                nn.Linear(self.window_size, latent_dim-nr_shared),
                nn.Tanh()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, self.window_size),
                nn.Tanh()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.window_size, intermediate_dim),
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
                nn.Linear(intermediate_dim, self.window_size),
                nn.Tanh()
            )
        self.apply(utils.weights_init)
            
    def encode(self, x):
        z = self.encoder(x)
        z_shared = self.encoder_shared(z)
        z_unshared = self.encoder_unshared(z)
        z = torch.cat([z_shared, z_unshared], -1)
        return z_shared, z_unshared, z
    
    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        z_shared, z_unshared, z = self.encode(x)
        x_decoded = self.decode(z)
        mse_loss = F.mse_loss(x, x_decoded)

        shared_loss = F.mse_loss(z_shared[:,1:,:],z_shared[:,:self.nr_ae-1,:])
        
        return mse_loss + self.loss_weight*shared_loss
    
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
                tbar.refresh() # to show immediately the update
        
    def encode_windows(self, windows):
        device = self.device
        new_windows = torch.from_numpy(self.prepare_input_windows(windows)).float().to(device)
        with torch.no_grad():
            _, _, encoded_windows_pae = self.encode(new_windows)
        encoded_windows_pae = encoded_windows_pae.detach().cpu().numpy()
        encoded_windows = np.concatenate((encoded_windows_pae[:,0,:self.nr_shared],
                                          encoded_windows_pae[-self.nr_ae+1:,self.nr_ae-1,:self.nr_shared]),axis=0)

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

class TIRE(nn.Module):
    def __init__(self, window_size:int=20, 
                 intermediate_dim_TD:int=0, 
                 intermediate_dim_FD:int=10,
                 nfft:int=30,
                 norm_mode:str='timeseries',
                 domain:str='both',
                 **kwargs):
        """
        Create a TIRE model.

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
        super().__init__()
        self.window_size_td = window_size
        self.AE_TD = AE(self.window_size_td, intermediate_dim=intermediate_dim_TD, **kwargs)
        self.window_size_fd = utils.calc_fft(np.random.randn(100, window_size), nfft, norm_mode).shape[-1]
        self.AE_FD = AE(self.window_size_fd, intermediate_dim=intermediate_dim_FD, **kwargs)
        
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
        shared_features_TD, shared_features_FD = [ae.encode_windows(windows) for ae, windows in [(self.AE_TD, windows_TD), (self.AE_FD, windows_FD)]]
        
        dissimilarities = smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, 
                                                                 self.domain, self.window_size_td)
        scores = change_point_score(dissimilarities, self.window_size_td)
        return dissimilarities, scores
    
    @staticmethod
    def ts_to_windows(ts, window_size):
        shape = ts.shape[:-1] + (ts.shape[-1] - window_size + 1, window_size)
        strides = ts.strides + (ts.strides[-1],)
        return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)
    

def smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, domain, window_size):
    """
    Calculation of smoothened dissimilarity measures
    
    Args:
        encoded_windows: TD latent representation of windows
        encoded_windows_fft:  FD latent representation of windows
        domain: TD/FD/both
        parameters: array with used parameters
        window_size: window size used
        par_smooth
        
    Returns:
        smoothened dissimilarity measures
    """
    
    if domain == "TD":
        encoded_windows_both = encoded_windows
    elif domain == "FD":
        encoded_windows_both = encoded_windows_fft
    elif domain == "both":
        beta = np.quantile(utils.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(utils.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows*alpha, encoded_windows_fft*beta),axis=1)
    
    encoded_windows_both = utils.matched_filter(encoded_windows_both, window_size)
    distances = utils.distance(encoded_windows_both, window_size)
    distances = utils.matched_filter(distances, window_size)
    
    return distances

def change_point_score(distances, window_size):
    """
    Gives the change point score for each time stamp. A change point score > 0 indicates that a new segment starts at that time stamp.
    
    Args:
    distances: postprocessed dissimilarity measure for all time stamps
    window_size: window size used in TD for CPD
        
    Returns:
    change point scores for every time stamp (i.e. zero-padded such that length is same as length time series)
    """
    
    prominences = np.array(utils.new_peak_prominences(distances)[0])
    prominences = prominences/np.amax(prominences)
    return np.concatenate((np.zeros((window_size,)), prominences, np.zeros((window_size-1,))))

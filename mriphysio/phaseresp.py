"""
Class for estimating the respiratory waveform from the phase channel of an EPI timeseries
- Developed specifically for the Dense Amygdala project, but more widely useful for complex-valued fMRI
- Required temporally unwrapped complex phase difference BOLD timeseries with critical or better temporal sampling of respiration
- Do NOT use Laplacian phase unwrapping which removes most of the respiration phase modulation

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center
DATES  : 2024-12-05 JMT Build from ideas in Calhoun complex phase fMRI
"""

import json
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
import scipy as sp
from sklearn.decomposition import PCA
from .bidsphysio import save_bids_physio


class PhaseRespWave:

    def __init__(self, dphi_pname, wmask_pname, method='weighted_mean', n_pca=1):
        """

        :param dphi_pname: str, Pathlike
            Path to the temporally unwrapped complex phase difference 3D x time Nifti image
        :param wmask_pname: str, Pathlike
            Path to weighting mask Nifti image. Contains floats [0, 1]
        :param method: str, optional
            Method for respiration waveform estimation ['weighted_mean' or 'pca']
        :param n_pca: int, optional
            Number of PCA components to retain [1]
        """

        # Load the 4D phase difference image
        self.dphi_nii = nib.load(dphi_pname)
        self.dphi = self.dphi_nii.get_fdata()

        # Create a time vector from the TR and number of timepoints
        self.tr_s = self.dphi_nii.header['pixdim'][4]
        self.nt = self.dphi.shape[3]

        # Time vector for mid-TR of each volume
        self.t_s = (np.arange(0, self.nt) + 0.5) * self.tr_s

        # Load the 3D probabilistic computation mask
        self.wmask_nii = nib.load(wmask_pname)
        self.w = self.wmask_nii.get_fdata()

        # Clamp weights to [0, 1]
        self.w = np.clip(self.w, 0, 1)

        # PCA components to retain
        self.n_pca = n_pca

        assert method in ['weighted_mean', 'pca'], f'Unknown method {method}'
        self.method = method

    def estimate(self):
        """
        Respiratory waveform estimation from phase difference 3D x time image
        """

        print('\nRespiration Waveform Reconstruction')
        print('-------------------------------------')
        print(f'  Method: {self.method}')
        if self.method == 'pca':
            print(f'  Number of PCA components: {self.n_pca}')
        print(f'  TR: {self.tr_s:.3f} s')
        print(f'  Number of timepoints: {self.nt}')
        print(f'  Voxel dimensions: {self.dphi.shape[0:3]}')

        # Flatten the spatial dimensions and transpose to create temporal-spatial data matrix X
        # X is Nt x Nvox
        X = self.dphi.reshape(-1, self.dphi.shape[3]).T

        match self.method:

            case 'weighted_mean':
                # Weighted mean of signal
                # Calculate the mask weighted spatial mean phase difference for all volumes
                mean_dphi = np.average(X, axis=1, weights=self.w.flatten())

                # High pass filter by subtracting savgol baseline
                # For similarity to typical first PCA component
                k = int(10.0 / self.tr_s)
                bline = sp.signal.savgol_filter(mean_dphi, window_length=k, polyorder=3)
                mean_dphi_hpf = mean_dphi - bline

                respwave = np.zeros([self.nt, 2])
                respwave_dict = {
                    'Time': self.t_s,
                    'Resp': mean_dphi_hpf,
                }
                self.resp_df = pd.DataFrame(respwave_dict)

                # Add peak inhalation/exhalation labels
                self.resp_peaks()

            case 'pca':
                # Weighted temporal PCA of signal
                # Returns temporal components and spatial weights
                temporal_components, spatial_weights, _ = self.weighted_pca(X, self.w.flatten(), self.n_pca)

                # Retain first component as respiration waveform
                respwave = np.zeros([self.nt, 1+self.n_pca])
                respwave[:, 0] = self.t_s
                respwave[:, 1:] = temporal_components

                # Construct respiration waveform dataframe
                resp_headers = [f'Resp{pc+1}' for pc in range(self.n_pca)]
                self.resp_df = pd.DataFrame(respwave, columns=['Time'] + resp_headers)

                # Add peak inhalation/exhalation labels
                self.resp_peaks()

        return self.resp_df
    
    def resp_peaks(self):
        """
        Find peak inhalation and exhalation in the respiration waveform
        """

        # Find positive peaks in respiration waveform
        min_period = 1.0 / self.tr_s
        pos_peaks, _ = sp.signal.find_peaks(self.resp_df['Resp'], distance=min_period)
        neg_peaks, _ = sp.signal.find_peaks(-self.resp_df['Resp'], distance=min_period)

        # Add peak locations to respiration waveform dataframe
        self.resp_df['Peak'] = 'n/a'
        self.resp_df.loc[pos_peaks, 'Peak'] = 'Inhalation'
        self.resp_df.loc[neg_peaks, 'Peak'] = 'Exhalation'

        return self.resp_df
    
    def to_bids(self, bids_outdir, bids_stub):
        
        # Save the respiration waveform to BIDS physio format (tsv.gz + json)
        save_bids_physio(bids_outdir, bids_stub, self.resp_df)
        
    @staticmethod
    def weighted_pca(X, w, n_pca=1):
        """
        Weighted PCA of temporal (Nt rows) x spatial (Nvox columns) data matrix X
        - spatial weights are applied to each column of X
        
        :param X: np.ndarray, shape (Nt, Nvox)
            Temporal spatial data matrix
        :param w: np.ndarray, shape (Nvox,)
            Voxel Weights for each column of X
        :param n_pca: int, optional (default = 1)
            Number of PCA components to retain

        """

        # Step 1: Apply spatial weights
        Xw = X * w  # Broadcasting

        # Step 2: Standardize the data (optional, depending on your needs)
        mean = np.mean(Xw, axis=0)
        sd = np.std(Xw, axis=0)
        Xstd = np.zeros_like(Xw)
        nonzero = sd > 0.0
        Xstd[:, nonzero] = (Xw[:, nonzero] - mean[nonzero]) / sd[nonzero]

        # Step 3: Perform PCA
        pca = PCA(n_components=n_pca)
        pca.fit(Xstd)

        # Report key PCA results
        print(f'  PCA singular values  : {pca.singular_values_}')
        print(f'  PCA exp var ratio    : {pca.explained_variance_ratio_}')
        
        # Step 4: Retrieve results
        temporal_components = pca.transform(Xstd)  # Principal components
        spatial_patterns = pca.components_  # Eigenvectors for spatial patterns
        
        return temporal_components, spatial_patterns, pca



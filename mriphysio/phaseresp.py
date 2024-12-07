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
from scipy.signal import spectrogram, find_peaks
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


class PhaseRespWave:

    def __init__(self, dphi_pname, wmask_pname, n_pca=1):
        """

        :param dphi_pname: str, Pathlike
            Path to the temporally unwrapped complex phase difference 3D x time Nifti image
        :param wmask_pname: str, Pathlike
            Path to weighting mask Nifti image. Contains floats [0, 1]
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
        self.n_pca = 1

    def estimate(self):
        """
        Respiratory waveform estimation from phase difference 3D x time image
        """

        print('\nRespiration Waveform Reconstruction')
        print('-------------------------------------')

        # Flatten the spatial dimensions and transpose to create temporal-spatial data matrix X
        # X is Nt x Nvox
        X = self.dphi.reshape(-1, self.dphi.shape[3]).T

        # Weighted temporal PCA of signal
        # Returns temporal components and spatial weights
        temporal_components, spatial_weights, _ = self.weighted_pca(X, self.w.flatten(), self.n_pca)

        # Retain first component as respiration waveform
        respwave = np.zeros([self.nt, 1+self.n_pca])
        respwave[:, 0] = self.t_s
        respwave[:, 1:] = temporal_components

        # Construct respiration waveform dataframe
        resp_headers = [f'Resp{pc+1}' for pc in range(self.n_pca)]
        self.resp_df = pd.DataFrame(respwave, columns=['Time (s)'] + resp_headers)

        return self.resp_df
        
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

    def to_bids(self, out_dir, bids_stub):

        # Sampling frequency from TR
        samp_freq = 1.0 / self.tr_s

        # Output physio TSV and JSON filenames
        tsv_fname = op.join(out_dir, f'{bids_stub}.tsv')
        json_fname = op.join(out_dir, f'{bids_stub}.json')

        # Save respiration waveform to BIDS physio TSV
        print(f'Saving respiratory waveform to {tsv_fname}')
        self.resp_df.to_csv(
            tsv_fname,
            sep='\t',
            index=False,
            float_format='%0.6f'
        )

        # Create dictionary of metadata and write to JSON sidecar
        json_dict = {
            "SamplingFrequency": samp_freq,
            "StartTime": self.resp_df['Time (s)'].min(),
            "Columns": self.resp_df.columns.tolist(),
            "Time": {
                "Units": "s"
            }
        }

        # Finally write JSON sidecar
        print(f'Saving JSON sidecar to {json_fname}')
        self._write_json(json_fname, json_dict, overwrite=True)

    @staticmethod
    def _write_json(json_fname, json_dict, overwrite=False):

        if op.isfile(json_fname):
            if overwrite:
                create_file = True
            else:
                create_file = False
        else:
            create_file = True

        if create_file:
            with open(json_fname, 'w') as fd:
                json.dump(json_dict, fd, indent=4, separators=(',', ':'))



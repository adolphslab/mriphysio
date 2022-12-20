"""
Class for opening and converting a DICOM physiology waveform object (CMRR) to TSV text file
"""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


class HeartRate:

    def __init__(self, t, ecg):
        """

        :param t: array, float
            Waveform time vector in seconds
        :param ecg: array, float
            nt x (1 .. 4) of ECG lead waveforms
        :param tsv_fname, str, path-like
            Output BIDS heartrate TSV filename
        """

        self._t = t
        self._ecg = ecg
        self._nt = len(t)
        self._nt_ecg = ecg.shape[0]
        self._n_leads = ecg.shape[1]

        # Dwell time and sampling frequency (Hz)
        self.dt = t[1] - t[0]
        self.fs = 1.0 / self.dt

        # Heart rate estimation window in seconds
        self._window = 10.0

        # Heart rate and associated time vector
        self.hr = pd.DataFrame()

        if not self._nt_ecg == self._nt:
            raise Exception(f'ECG waveforms and time vector sample mismatch ({self._nt_ecg} vs {self._nt})')

    def estimate(self):
        """
        Heart rate estimation from CMRR Physiolog ECG
        - R-wave timing difficult to estimate from magnetohemodynamic contaminated ECG waveforms
        - Moving FFT (spectrogram) analysis with peak tracking in typical HR frequency range

        AUTHOR : Mike Tyszka
        PLACE  : Caltech Brain Imaging Center
        DATES  : 2022-12-20 JMT Build on mriphysio package
        """

        print('\nHeart Rate Estimation')
        print('---------------------')

        # Grab Lead I (actual lead V3)
        ecg1 = self._ecg[:, 0]

        # Construct spectrogram with sufficient frequency resolution (< 0.5Hz)
        f_spec, t_spec, sxx = spectrogram(
            ecg1,
            fs=self.fs,
            nperseg=int(self._window / self.dt),
            nfft=8000
        )

        print(f'Sampling frequency : {self.fs} Hz')
        print(f'Temporal samples   : {len(ecg1)}')
        print(f'Sampling time      : {np.max(t_spec)} s')
        print(f'Temporal bins      : {len(t_spec)}')
        print(f'Temporal res       : {t_spec[1] - t_spec[0]} s')
        print(f'Frequency bins     : {len(f_spec)}')
        print(f'Frequency res      : {f_spec[1] - f_spec[0]} Hz')

        # Crop spectrogram frequencies to capture physiological HR range (0.5 to 3.5 Hz) and
        # RF pulse interference (approx 12.5 Hz)
        f_max = 20.0
        mask = f_spec <= f_max
        f_crop = f_spec[mask]
        sxx_crop = sxx[mask, :]

        # Binarize spectrogram (10% max)
        p_thresh = np.max(sxx_crop) / 10.0
        sxx_bin = (sxx_crop > p_thresh).astype(float)

        # Project binarized spectrogram in time
        p_tmax = np.max(sxx_bin, axis=1)

        # Plot spectrogram and temporal mean power
        # fig, axs = plt.subplots(1, 2, width_ratios=[3, 1], figsize=(16, 8), sharey=True)
        # axs[0].pcolormesh(ts, f_crop, sxx_bin)
        # axs[0].set_ylabel('Frequency [Hz]')
        # axs[0].set_xlabel('Time [sec]')
        #
        # axs[1].plot(p_tmax, f_crop)
        # plt.show()

        # Locate fundamental frequency band in projection
        band_indices = np.nonzero(p_tmax)[0]
        index_jumps = (np.diff(band_indices) > 1).astype(int)
        band_ends = np.nonzero(index_jumps)

        f0_start = band_indices[0]
        f0_end = band_indices[band_ends[0][0]] + 1

        print(f'Fundamental band   : [{f_spec[f0_start]:0.3f}, {f_spec[f0_end]:0.3f}) Hz')

        # Calculate weighted mean fundamental frequency from spectrogram columns

        # Carve out fundamental band in spectrogram
        Sxx_f0 = sxx_crop[f0_start:f0_end, :]
        f_f0 = f_crop[f0_start:f0_end]

        # Fundamental frequency from power-weighted mean frequency
        hr = np.matmul(Sxx_f0.transpose(), f_f0) / np.sum(Sxx_f0, axis=0)

        # fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        # axs.pcolormesh(t_spec, f_crop, sxx_crop)
        # axs.set_ylabel('Frequency [Hz]')
        # axs.set_xlabel('Time [sec]')
        # axs.plot(t_spec, hr, 'w')
        # axs.set_ylim(0, 3.5)
        # plt.show()

        # Construct heart rate dataframe
        hr_array = np.vstack([t_spec, hr]).transpose()
        self.hr = pd.DataFrame(hr_array, columns=['Time (s)', 'Heart Rate (Hz)'])

        return self.hr

    def save(self, hr_tsv):

        print(f'\nSaving estimated heart rate to {hr_tsv}')

        self.hr.to_csv(
            hr_tsv,
            sep='\t',
            index=False,
            float_format='%0.6f'
        )




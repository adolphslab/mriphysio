"""
Class for opening and converting a DICOM physiology waveform object (CMRR) to TSV text file
"""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, find_peaks
from scipy.interpolate import interp1d

# import matplotlib.pyplot as plt


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

        # Spectrogram temporal segment duration (s)
        self._t_seg = 10.0

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

        # ECG waveform duration (s)
        t_ecg = np.max(self._t)

        # Spectrogram temporal segment duration (s)
        t_seg = 10.0

        if t_ecg < t_seg:
            raise Exception(f'ECG waveform too short ({t_ecg} s) - need at least {t_seg} s')

        # Number of temporal samples per segment
        nps = int(self._t_seg / self.dt)

        freq_bw = self.fs / 2.0
        targ_freq_res = 0.05
        nfft = int(freq_bw / targ_freq_res * 2.0)

        print(f'ECG waveform dur   : {t_ecg} s')
        print(f'Sampling frequency : {self.fs} Hz')

        print(f'\nSpectrogram')
        print(f'-----------')
        print(f'Temporal samples   : {len(ecg1)}')
        print(f'Segment length     : {nps}')
        print(f'Frequency BW       : {freq_bw} Hz')
        print(f'Target freq res    : {targ_freq_res} Hz')
        print(f'FFT length         : {nfft}')

        # Construct spectrogram
        f_spec, t_spec, sxx = spectrogram(
            ecg1,
            fs=self.fs,
            nperseg=nps,
            nfft=nfft
        )

        dt_spec = t_spec[1] - t_spec[0]
        df_spec = f_spec[1] - f_spec[0]

        print(f'Temporal bins      : {len(t_spec)}')
        print(f'Temporal res       : {dt_spec} s')
        print(f'Frequency bins     : {len(f_spec)}')
        print(f'Frequency res      : {df_spec} Hz')

        # Crop spectrogram frequencies to capture physiological HR range (0.5 to 3.5 Hz) and
        # RF pulse interference (approx 12.5 Hz)
        f_min, f_max = 0.5, 15.0
        mask = (f_min <= f_spec) & (f_spec <= f_max)
        f_crop = f_spec[mask]
        sxx_crop = sxx[mask, :]

        # Temporal median spectral power
        # Robust identification of cardiac fundamental frequency over time
        p_tmed = np.median(sxx_crop, axis=1)

        i_peaks = find_peaks(p_tmed, distance=f_min / df_spec)
        f_peaks = f_crop[i_peaks[0]]
        print(f'Frequency peaks    :')
        print(f'  {f_peaks} Hz')

        # Find peak in temporal median power closest to 75 bpm (1.25 Hz)
        f0_targ = 1.25
        cardiac_peak = np.argmin(np.abs(f_peaks - f0_targ))
        f0 = f_peaks[cardiac_peak]
        print(f'Cardiac cycle peak : {f0:0.3f} Hz')

        # Create 75% to 150% frequency band around f0
        f0_low = f0 * 0.75
        f0_high = f0 * 1.50

        print(f'Cardiac cycle band : {f0_low:0.3f} to {f0_high:0.3f} Hz')

        # Find indices closest to heart band limits
        interpolator = interp1d(f_crop, np.arange(len(f_crop)), kind='nearest')
        i_low, i_high = interpolator([f0_low, f0_high]).astype(int)

        # Calculate weighted mean fundamental frequency from spectrogram columns

        # Carve out fundamental band in spectrogram
        sxx_f0 = sxx_crop[i_low:i_high, :]
        f_f0 = f_crop[i_low:i_high]

        # Fundamental frequency from power-weighted mean frequency
        f0 = np.matmul(sxx_f0.transpose(), f_f0) / np.sum(sxx_f0, axis=0)

        # fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        # axs.pcolormesh(t_spec, f_crop, np.log(sxx_crop))
        # axs.set_ylabel('Frequency [Hz]')
        # axs.set_xlabel('Time [sec]')
        # axs.plot(t_spec, f0, 'white', linewidth=2)
        # axs.set_ylim(0, 5)
        # plt.show()

        # ECG waveform quality control
        # Calculate weighted sd of frequency in cardiac fundamental band.
        print('\nRunning quality control on ECG waveform')

        # Sum over frequency of fundamental band spectrogram
        # Use for weight normalization
        sxx_f0_fsum = np.sum(sxx_f0, axis=0)

        # Power-weighted SD over frequency
        wsd = np.zeros_like(t_spec)
        for tc, tt in enumerate(t_spec):
            wsd[tc] = np.sqrt(np.sum(sxx_f0[:, tc] * (f_f0 - f0[tc]) ** 2) / sxx_f0_fsum[tc])

        # Weighted SDs > 0.15 Hz generally indicate poor ECG quality in that time bin
        qc_pass = wsd < 0.15
        print(f'Quality control pass rate : {np.sum(qc_pass) / len(qc_pass) * 100.0:0.1f} %')

        # Construct heart rate dataframe
        hr_array = np.vstack([t_spec, f0, wsd]).transpose()
        self.hr = pd.DataFrame(hr_array, columns=['Time (s)', 'Heart Rate (Hz)', 'wSD (Hz)'])

        return self.hr

    def save(self, hr_tsv):

        print(f'\nSaving estimated heart rate to {hr_tsv}')

        self.hr.to_csv(
            hr_tsv,
            sep='\t',
            index=False,
            float_format='%0.6f'
        )




"""
Class for opening and converting a DICOM physiology waveform object (CMRR) to TSV text file
"""

import os.path as op
import sys
import pydicom
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.interpolate import interp1d


class ECG:

    def __init__(self, t, ecg):
        """

        :param t: array, float
            Waveform time vector in seconds
        :param ecg: array, float
            nt x (1 .. 4) of ECG lead waveforms
        """

        self._t = t
        self._ecg = ecg
        self._nt = len(t)
        self._nt_ecg = ecg.shape[0]
        self._n_leads = ecg.shape[1]

        if not self._nt_ecg == self._nt:
            raise Exception(f'ECG waveforms and time vector sample mismatch ({self._nt_ecg} vs {self._nt})')





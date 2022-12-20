"""
Class for opening and converting a DICOM physiology waveform object (CMRR) to TSV text file
"""

import os.path as op
import sys
import pydicom
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class PhysioDicom:

    def __init__(self, physio_dcm):

        print(f'Opening {op.realpath(physio_dcm)}')

        assert op.isfile(physio_dcm), f'* {physio_dcm} does not exist'

        # Init waveform dataframe
        self.waveforms = pd.DataFrame()

        # Channel flags
        self.have_ecg = False
        self.have_resp = False
        self.have_puls = False

        # Create DICOM object
        self._physio_dcm = physio_dcm
        self._dcm = pydicom.dcmread(physio_dcm, stop_before_pixels=True)

        # Extract data from Siemens spectroscopy tag (0x7fe1, 0x1010)
        # Yields one long byte array
        self._physio_data = self._dcm[0x7fe1, 0x1010].value

        # Extract relevant info from header
        self.n_points = len(self._physio_data)
        self.n_rows = self._dcm.AcquisitionNumber

        # Presumptive sampling time for one tick @ 400 Hz (2.5 ms)
        # Siemens Prisma VE11C, DICOM Physio Logging
        self.dt = 2.5e-3
        print(f'Presumptive sample time per tick : {self.dt * 1000.0:0.1f} ms')

        if self.n_points % self.n_rows:
            print(f'* Points ({self.n_points}) is not an integer multiple of rows ({self.n_rows}) - exiting')
            sys.exit(-1)

        self.n_cols = int(self.n_points / self.n_rows)

        if (self.n_points % 1024):
            print(f'* Columns ({self.n_cols}) is not an integer multiple of 1024 - exiting')
            sys.exit(-1)

        self.n_waves = int(self.n_cols / 1024)
        self.wave_len = int(self.n_points / self.n_waves)

        print('')
        print(f'Detected {self.n_points} points in {self.n_rows} rows and {self.n_cols} columns')
        print(f'Detected {self.n_waves} waveforms')

    def convert(self):

        # Init time and waveform vectors
        t_ecg, t_puls, t_resp = [], [], []
        s_ecg1, s_ecg2, s_ecg3, s_ecg4 = [], [], [], []
        s_puls, s_resp = [], []
        tick_ecg, tick_puls, tick_resp = -1.0, -1.0, -1.0

        # Parse each waveform type (ECG, pulse, resp) separately
        for wc in range(self.n_waves):

            print('')
            print(f'Parsing waveform {wc}')

            offset = wc * self.wave_len

            wave_data = self._physio_data[slice(offset, offset+self.wave_len)]

            data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
            fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
            fname = wave_data[slice(8, 8+fname_len)]

            print(f'Data length     : {data_len}')
            print(f'Filename length : {fname_len}')
            print(f'Filename        : {fname}')

            # Extract waveform log byte data
            log_bytes = wave_data[slice(1024, 1024+data_len)]

            # Parse waveform log
            waveform_name, t, s, tick = self._parse_log(log_bytes)

            if "ECG" in waveform_name:
                t_ecg, tick_ecg = t, tick
                s_ecg1, s_ecg2, s_ecg3, s_ecg4 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

            if "PULS" in waveform_name:
                t_puls, s_puls, tick_puls = t, s, tick

            if "RESP" in waveform_name:
                t_resp, s_resp, tick_resp = t, s, tick

        # Set availability flag for each waveform
        self.have_ecg = len(t_ecg) > 0
        self.have_resp = len(t_resp) > 0
        self.have_puls = len(t_puls) > 0

        # Zero time origin (identical offset for all waveforms) and scale to seconds
        if self.have_ecg:
            t_ecg = (t_ecg - t_ecg[0]) * tick_ecg * self.dt
        if self.have_puls:
            t_puls = (t_puls - t_puls[0]) * tick_puls * self.dt
        if self.have_resp:
            t_resp = (t_resp - t_resp[0]) * tick_resp * self.dt

        # Create master clock vector in seconds from highest sampling rate time vector
        # Typically this is the ECG time vector if acquired
        if self.have_ecg:
            t_master = t_ecg
        elif self.have_puls:
            t_master = t_puls
        elif self.have_resp:
            t_master = t_resp
        else:
            print(f'* No waveform time vector found - exiting')
            sys.exit(1)

        # Resample all waveforms to master clock vector
        print('')
        print('Interpolating waveforms to master clock')

        # Construct dataframe for available interpolated waveforms
        # Init with master time vector
        self.waveforms = pd.DataFrame({'Time_s': t_master})

        if self.have_ecg:
            self.waveforms['ECG1'] = self._resample(t_master, t_ecg, s_ecg1)
            self.waveforms['ECG2'] = self._resample(t_master, t_ecg, s_ecg2)
            self.waveforms['ECG3'] = self._resample(t_master, t_ecg, s_ecg3)
            self.waveforms['ECG4'] = self._resample(t_master, t_ecg, s_ecg4)

        if self.have_puls:
            self.waveforms['Pulse'] = self._resample(t_master, t_puls, s_puls)

        if self.have_resp:
            self.waveforms['Resp'] = self._resample(t_master, t_resp, s_resp)

        return self.waveforms

    def save(self, physio_tsv=None):

        if not physio_tsv:
            physio_tsv = self._physio_dcm.replace('.dcm', '.tsv')

        print(f'\nSaving pulse and respiratory waveforms to {physio_tsv}')

        self.waveforms.to_csv(physio_tsv,
                              sep='\t',
                              index=False,
                              float_format='%0.6f')

    def _parse_log(self, log_bytes):

        # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
        physio_string = log_bytes.decode('utf-8', 'ignore')

        # CMRR DICOM physio log has \n separated lines
        physio_lines = physio_string.splitlines()

        # Init parameters and lists
        uuid = "UNKNOWN"
        data_type = "UNKNOWN"
        scan_date = "UNKNOWN"
        tick = -1
        t_list = []
        s_list = []
        ch_list = []

        for line in physio_lines:

            # Divide the line at whitespace
            parts = line.split()

            # Data lines have the form "<tag> = <value>" or "<time> <channel> <signal>"
            if len(parts) == 3:

                p1, p2, p3 = parts

                if 'UUID' in p1:
                    uuid = p3

                if 'ScanDate' in p1:
                    scan_date = p3

                if 'LogDataType' in p1:
                    data_type = p3

                if 'SampleTime' in p1:
                    tick = int(p3)

                if 'ECG' in p2 or 'PULS' in p2 or 'RESP' in p2:
                    t_list.append(float(p1))
                    s_list.append(float(p3))
                    ch_list.append(p2)

        print(f'UUID            : {uuid}')
        print(f'Scan date       : {scan_date}')
        print(f'Data type       : {data_type}')
        print(f'Tick interval   : {tick:d}')

        # Convert to numpy arrays
        t, s, ch = np.array(t_list), np.array(s_list), np.array(ch_list)

        if 'ECG' in data_type:

            # Separate timestamps and signal for each ECG channel (1..4)
            i1 = ch == 'ECG1'
            i2 = ch == 'ECG2'
            i3 = ch == 'ECG3'
            i4 = ch == 'ECG4'

            # In general t1..t4 are different lengths but with same min, max
            # Assuming some values are dropped. Time vectors have unit spacing
            t1, s1 = t[i1], s[i1]
            t2, s2 = t[i2], s[i2]
            t3, s3 = t[i3], s[i3]
            t4, s4 = t[i4], s[i4]

            # Remove duplicate time points (typically only one or two duplicates in a waveform)
            t1, s1 = self._remove_duplicates(t1, s1)
            t2, s2 = self._remove_duplicates(t2, s2)
            t3, s3 = self._remove_duplicates(t3, s3)
            t4, s4 = self._remove_duplicates(t4, s4)

            # Find limits over all time vectors
            t_min = np.min([t1.min(), t2.min(), t3.min(), t4.min()])
            t_max = np.max([t1.max(), t2.max(), t3.max(), t4.max()])

            # Create new t vector
            t_i = np.arange(t_min, t_max+1)

            # Interpolate all ECG signals to new t vector
            f1 = interp1d(t1, s1, kind='cubic', fill_value='extrapolate')
            s1_i = f1(t_i)
            f2 = interp1d(t2, s2, kind='cubic', fill_value='extrapolate')
            s2_i = f2(t_i)
            f3 = interp1d(t3, s3, kind='cubic', fill_value='extrapolate')
            s3_i = f3(t_i)
            f4 = interp1d(t4, s4, kind='cubic', fill_value='extrapolate')
            s4_i = f4(t_i)

            # Stack channel signals into nt x 4 array
            t = t_i
            s = np.column_stack([s1_i, s2_i, s3_i, s4_i])

        return data_type, t, s, tick

    @staticmethod
    def _remove_duplicates(t, s):
        u, c = np.unique(t, return_counts=True)
        inds = np.where(c > 1)
        return np.delete(t, inds), np.delete(s, inds)

    @staticmethod
    def _resample(t_master, t, s):
        f = interp1d(t, s, kind='cubic', fill_value='extrapolate')
        return f(t_master)
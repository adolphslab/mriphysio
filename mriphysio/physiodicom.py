"""
Class for opening and converting a CMRR DICOM physiology log object to a TSV text file

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center

REFERENCES:
Original CMRR Matlab log parsing script
    https://github.com/CMRR-C2P/MB/blob/master/extractCMRRPhysio.m
David Warren's summary of CMRR Physiolog contents
    https://david-e-warren.me/blog/physiological-data-from-mri/

LICENSE: MIT Copyright Caltech 2023
"""

import os.path as op
import sys
import pydicom
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class PhysioDicom:

    def __init__(self, physio_dcm):

        print(f'Opening {op.realpath(physio_dcm)}\n')

        assert op.isfile(physio_dcm), f'* {physio_dcm} does not exist'

        # Init wave_lines dataframe
        self.waveforms = pd.DataFrame()

        # Channel presence flags
        self.have_ecg = False
        self.have_resp = False
        self.have_puls = False
        self.have_acq = False

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

        print(f'Detected {self.n_waves} waveforms')
        print(f'Detected {self.n_points} points in {self.n_rows} rows and {self.n_cols} columns')

    def convert(self):

        # Init time and wave_lines vectors
        t_ecg, t_puls, t_resp, t_acq = [], [], [], []
        s_ecg1, s_ecg2, s_ecg3, s_ecg4 = [], [], [], []
        s_puls, s_resp, s_acq = [], [], []

        # Parse each wave_lines type (ECG, pulse, resp, EPI info) separately
        for wc in range(self.n_waves):

            print('')
            print(f'Parsing wave_lines {wc}')

            offset = wc * self.wave_len
            wave_data = self._physio_data[slice(offset, offset+self.wave_len)]

            data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
            fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
            fname = wave_data[slice(8, 8+fname_len)]

            print(f'Data length     : {data_len}')
            print(f'Filename length : {fname_len}')
            print(f'Filename        : {fname}')

            # Extract wave_lines log byte data
            log_bytes = wave_data[slice(1024, 1024+data_len)]

            # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
            # CMRR DICOM physio log has \n separated lines - use splitlines() to separate into list
            physio_lines = log_bytes.decode('utf-8', 'ignore').splitlines()

            # Parse log contents for this wave_lines
            # hdr will contain additional keys depending on wave_lines type
            hdr, t, s = self._parse_log(physio_lines)
            data_type = hdr['DataType']

            print(f"Data Type       : {hdr['DataType']}")

            if "ECG" in data_type:
                t_ecg = t
                s_ecg1, s_ecg2, s_ecg3, s_ecg4 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

            elif "PULS" in data_type:
                t_puls, s_puls = t, s

            elif "RESP" in data_type:
                t_resp, s_resp = t, s

            elif 'ACQUISITION_INFO' in data_type:
                t_acq, s_acq = t, s

        # Set availability flag for each wave_lines
        self.have_ecg = len(t_ecg) > 0
        self.have_resp = len(t_resp) > 0
        self.have_puls = len(t_puls) > 0
        self.have_acq = len(t_acq) > 0

        # Zero time origin (identical offset for all waveforms) and scale to seconds
        # Note that raw time vectors are in ticks (2.5 ms) and don't need further scaling by the reported
        # tick interval, just scaling by self.dt (2.5 ms)
        if self.have_ecg:
            t_ecg_s = (t_ecg - t_ecg[0]) * self.dt
        if self.have_puls:
            t_puls_s = (t_puls - t_puls[0]) * self.dt
        if self.have_resp:
            t_resp_s = (t_resp - t_resp[0]) * self.dt
        if self.have_acq:
            t_acq_s = (t_acq - t_acq[0]) * self.dt

        # Create master clock vector in seconds from highest sampling rate time vector
        # Typically this is the acquisition window time vector if present
        if self.have_acq:
            t_master = t_acq_s
        elif self.have_ecg:
            t_master = t_ecg_s
        elif self.have_puls:
            t_master = t_puls_s
        elif self.have_resp:
            t_master = t_resp_s
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
            self.waveforms['ECG1'] = self._resample(t_master, t_ecg_s, s_ecg1)
            self.waveforms['ECG2'] = self._resample(t_master, t_ecg_s, s_ecg2)
            self.waveforms['ECG3'] = self._resample(t_master, t_ecg_s, s_ecg3)
            self.waveforms['ECG4'] = self._resample(t_master, t_ecg_s, s_ecg4)

        if self.have_puls:
            self.waveforms['PULS'] = self._resample(t_master, t_puls_s, s_puls)

        if self.have_resp:
            self.waveforms['RESP'] = self._resample(t_master, t_resp_s, s_resp)

        if self.have_acq:
            self.waveforms['ACQ'] = self._resample(t_master, t_acq_s, s_acq)

        return self.waveforms

    def save(self, physio_tsv=None):

        # Create TSV filename if none provided
        if not physio_tsv:
            physio_tsv = self._physio_dcm.replace('.dcm', '.tsv')

        print(f'\nSaving physiological waveforms to {physio_tsv}')
        self.waveforms.to_csv(physio_tsv,
                              sep='\t',
                              index=False,
                              float_format='%0.6f')

    def _parse_log(self, physio_lines):

        # Common header keys for all waveforms
        hdr = {
            'UUID': self._parse_line(physio_lines[0])[1],
            'ScanDate': self._parse_line(physio_lines[1])[1],
            'LogVersion': self._parse_line(physio_lines[2])[1],
            'DataType': self._parse_line(physio_lines[3])[1]
        }

        # Waveform-specific header keys
        match hdr['DataType']:
            case 'PULS' | 'RESP' | 'ECG':
                hdr['SampTimeSecs'] = int(self._parse_line(physio_lines[4])[1]) * self.dt
                hdr['SampRateHz'] = 1.0 / hdr['SampTimeSecs']
                wave_start_line = 8
            case 'ACQUISITION_INFO':
                hdr['NumSlices'] = int(self._parse_line(physio_lines[4])[1])
                hdr['NumVolumes'] = int(self._parse_line(physio_lines[5])[1])
                hdr['NumEchoes'] = int(self._parse_line(physio_lines[6])[1])
                wave_start_line = 10
            case _:
                wave_start_line = 0
                pass

        # Isolate lines containing only wave_lines information
        wave_lines = physio_lines[wave_start_line:]

        # Parse wave_lines tic timestamps and signal
        t, s = [], []

        match hdr['DataType']:
            case 'PULS' | 'RESP':
                t, s = self._parse_pulsresp(wave_lines)
            case 'ECG':
                t, s = self._parse_ecg(wave_lines)
            case 'ACQUISITION_INFO':
                t, s = self._parse_acq_info(wave_lines)
            case '_':
                pass

        return hdr, t, s

    def _parse_line(self, line):

        # Divide the line at whitespace
        parts = line.split()

        if len(parts) == 3:
            tag, _, val = parts
        else:
            tag, val = 'UNKNOWN', 'UNKNOWN'

        return tag, val

    def _parse_pulsresp(self, wave_lines):
        """
        Waveform parser for PULS and RESP data types

        :param wave_lines: list, str
        :return: t, s: array, float
        """

        # Init lists
        t_list, s_list = [], []

        for line in wave_lines:

            # Waveform lines have form <time> <channel> <signal> [<trigger flag>]
            # Divide the line at whitespace and take parts 0 and 2
            parts = line.split()
            tick, signal = parts[0], parts[2]
            t_list.append(float(tick))
            s_list.append(float(signal))

        # Convert to numpy arrays and return
        return np.array(t_list), np.array(s_list)

    def _parse_ecg(self, wave_lines):
        """
        Waveform parser for PULS and RESP data types

        :param wave_lines: list, str
        :return: t, s: array, float
        """

        # Init lists
        t_list, ch_list, s_list = [], [], []

        for line in wave_lines:

            # Waveform lines have form <time> <channel> <signal> [<trigger flag>]
            # Divide the line at whitespace
            parts = line.split()
            tick, channel, signal = parts[0:3]
            t_list.append(float(tick))
            ch_list.append(channel)
            s_list.append(float(signal))

        # Convert to numpy arrays
        t, ch, s = np.array(t_list), np.array(ch_list), np.array(s_list)

        # Post process four-channel ECG waveforms
        # Create index masks for each ECG channel
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

        # Remove duplicate time points (typically only one or two duplicates in a wave_lines)
        t1, s1 = self._remove_duplicates(t1, s1)
        t2, s2 = self._remove_duplicates(t2, s2)
        t3, s3 = self._remove_duplicates(t3, s3)
        t4, s4 = self._remove_duplicates(t4, s4)

        # Find limits over all time vectors
        t_min = np.min([t1.min(), t2.min(), t3.min(), t4.min()])
        t_max = np.max([t1.max(), t2.max(), t3.max(), t4.max()])

        # Create new t vector
        t_i = np.arange(t_min, t_max + 1)

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

        return t, s

    def _parse_acq_info(self, wave_lines):
        """
        Waveform parser for ACQUISITION_TIME data type

        Unlike the physio waveforms, each acq info line has the form

        :param wave_lines: list, str
        :return: t, s: array, float
        """

        # Init acq window list
        acq_win_list = []

        for line in wave_lines:

            # Acquisition info lines have form:
            # VOLUME SLICE ACQ_START_TICS ACQ_FINISH_TICS ECHO
            # Divide the line at whitespace
            parts = line.split()
            if len(parts) == 5:
                acq_start, acq_finish = parts[2], parts[3]
                acq_win_list.append([int(acq_start), int(acq_finish)])

        # Convert acquisition start/end times to Nacq x 2 array
        acq_win = np.array(acq_win_list).astype(int)
        t_min, t_max = np.min(acq_win), np.max(acq_win)

        # Add a short preamble before the initial acquisition window
        t_min = t_min - 10

        # Create time and acquisition vectors
        t = np.arange(t_min, t_max+1)
        s = np.zeros_like(t)

        for acq in acq_win:
            t0, t1 = acq[0], acq[1]
            i0, i1 = t0 - t_min, t1 - t_min
            s[i0:i1] = 1

        return t, s

    @staticmethod
    def _remove_duplicates(t, s):
        u, c = np.unique(t, return_counts=True)
        inds = np.where(c > 1)
        return np.delete(t, inds), np.delete(s, inds)

    @staticmethod
    def _resample(t_master, t, s):
        f = interp1d(t, s, kind='cubic', fill_value='extrapolate')
        return f(t_master)

    def save_preview(self, preview_fname):
        """
        Save a preview plot of the first 30 seconds of all waveforms present

        :param preview_fname: str, pathlike
        :return:
        """

        print(f'Saving waveform preview to {preview_fname}')

        # Construct waveform list
        wave_list = []
        if self.have_acq:
            wave_list.append('ACQ')
        if self.have_ecg:
            wave_list.append('ECG1')
            wave_list.append('ECG2')
            wave_list.append('ECG3')
            wave_list.append('ECG4')
        if self.have_puls:
            wave_list.append('PULS')
        if self.have_resp:
            wave_list.append('RESP')

        n_wave = len(wave_list)
        t = self.waveforms['Time_s'].values

        i_crop = t < 10.0
        t_crop = t[i_crop]

        # Create a figure
        fig, axs = plt.subplots(n_wave, 1, figsize=(10, 6))

        for wc, wave in enumerate(wave_list):

            s = self.waveforms[wave].values
            s_crop = s[i_crop]

            axs[wc].plot(t_crop, s_crop)
            axs[wc].set_title(wave)

        plt.tight_layout()
        plt.savefig(preview_fname, dpi=300)
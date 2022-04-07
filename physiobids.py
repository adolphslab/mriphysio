#!/usr/bin/env python
"""
Purpose
----
Batch convert a directory of CMRR physiolog DICOM files to BIDS format and populate
a subject/session directory tree.

- Adapt parsing logic from extractCMRRPhysio.m (Ed Auerbach)
- Wrap previous single-file dcm2physio.py script in this repo

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

References
----
Matlab parser: https://github.com/CMRR-C2P/MB/blob/master/extractCMRRPhysio.m

License
----
MIT License

Copyright (c) 2021 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '0.1'


import os
import sys
import argparse
import pandas as pd
import numpy as np
import pydicom
from scipy.interpolate import interp1d


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DICOM files to BIDS-compliant Nifty structure')
    parser.add_argument('-i', '--infile', required=True, help='CMRR physio DICOM file')
    args = parser.parse_args()

    physio_dcm = args.infile

    # Create DICOM object
    d = pydicom.dcmread(physio_dcm, stop_before_pixels=True)

    # Extract data from Siemens spectroscopy tag (0x7fe1, 0x1010)
    # Yields one long byte array
    physio_data = d[0x7fe1, 0x1010].value

    # Extract relevant info from header
    n_points = len(physio_data)
    n_rows = d.AcquisitionNumber

    if n_points % n_rows:
        print('* Points (%d) is not an integer multiple of rows (%d) - exiting' % (n_points, n_rows))
        sys.exit(-1)

    n_cols = int(n_points / n_rows)

    if (n_points % 1024):
        print('* Columns (%d) is not an integer multiple of 1024 (%d) - exiting' % (n_cols, 1024))
        sys.exit(-1)

    n_waves = int(n_cols / 1024)
    wave_len = int(n_points / n_waves)

    print('')
    print(f'Detected {n_points} points in {n_rows} rows and {n_cols} columns')
    print(f'Detected {n_waves} waveforms')

    # Init time and waveform vectors
    t_ecg1, s_ecg1, dt_ecg1 = [], [], 1.0
    t_ecg2, s_ecg2, dt_ecg2 = [], [], 1.0
    t_ecg3, s_ecg3, dt_ecg3 = [], [], 1.0
    t_ecg4, s_ecg4, dt_ecg4 = [], [], 1.0
    t_puls, s_puls, dt_puls = [], [], 1.0
    t_resp, s_resp, dt_resp = [], [], 1.0

    # Parse each waveform type (ECG, pulse, resp) separately
    for wc in range(n_waves):

        print('')
        print('Parsing waveform %d' % wc)

        offset = wc * wave_len

        wave_data = physio_data[slice(offset, offset+wave_len)]

        data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
        fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
        fname = wave_data[slice(8, 8+fname_len)]

        print(f'Data length     : {data_len}')
        print(f'Filename length : {fname_len}')
        print(f'Filename        : {fname}')

        # Extract waveform log byte data
        log_bytes = wave_data[slice(1024, 1024+data_len)]

        # Parse waveform log
        waveform_name, t, s, dt = parse_log(log_bytes)

        if "ECG" in waveform_name:
            t_ecg, dt_ecg = t, dt
            s_ecg1, s_ecg2, s_ecg3, s_ecg4 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

        if "PULS" in waveform_name:
            t_puls, s_puls, dt_puls = t, s, dt

        if "RESP" in waveform_name:
            t_resp, s_resp, dt_resp = t, s, dt

    # Zero time origin (identical offset for all waveforms) and scale to seconds
    t_ecg = (t_ecg - t_ecg[0]) * dt_ecg
    t_puls = (t_puls - t_puls[0]) * dt_puls
    t_resp = (t_resp - t_resp[0]) * dt_resp

    print('')

    # Resample pulse waveform to match ECG waveform timing
    print('Interpolating pulse waveform to ECG sampling')
    f = interp1d(t_puls, s_puls, kind='cubic', fill_value='extrapolate')
    s_puls_i = f(t_ecg)

    # Resample respiration waveform to match ECG waveform timing
    print('Interpolating respiration waveform to ECG sampling')
    f = interp1d(t_resp, s_resp, kind='cubic', fill_value='extrapolate')
    s_resp_i = f(t_ecg)

    # Create a dataframe from a data dictionary
    print('Creating pandas dataframe')
    d = {
        'Time_s': t_ecg,
        'ECG1': s_ecg1,
        'ECG2': s_ecg2,
        'ECG3': s_ecg3,
        'ECG4': s_ecg4,
        'Pulse': s_puls_i,
        'Resp': s_resp_i
    }
    df = pd.DataFrame(d)

    # Export pulse and respiratory waveforms to TSV file
    tsv_fname = f"{os.path.splitext(physio_dcm)[0]}.tsv"

    print('')
    print('Saving pulse and respiratory waveforms to %s' % tsv_fname)

    df.to_csv(tsv_fname,
              sep='\t',
              columns=['Time_s', 'ECG1', 'ECG2', 'ECG3', 'ECG4', 'Pulse', 'Resp'],
              index=False,
              float_format='%0.3f')


def parse_log(log_bytes):

    # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
    physio_string = log_bytes.decode('utf-8', 'ignore')

    # CMRR DICOM physio log has \n separated lines
    physio_lines = physio_string.splitlines()

    # Init parameters and lists
    uuid = "UNKNOWN"
    data_type = "UNKNOWN"
    scan_date = "UNKNOWN"
    dt = -1.0
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
                dt = float(p3) * 1e-3

            if 'ECG' in p2 or 'PULS' in p2 or 'RESP' in p2:
                t_list.append(float(p1))
                s_list.append(float(p3))
                ch_list.append(p2)

    print(f'UUID            : {uuid}')
    print(f'Scan date       : {scan_date}')
    print(f'Data type       : {data_type}')

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
        t1, s1 = remove_duplicates(t1, s1)
        t2, s2 = remove_duplicates(t2, s2)
        t3, s3 = remove_duplicates(t3, s3)
        t4, s4 = remove_duplicates(t4, s4)

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

    # Return numpy arrays
    return data_type, t, s, dt


def remove_duplicates(t, s):
    u, c = np.unique(t, return_counts=True)
    inds = np.where(c > 1)
    return np.delete(t, inds), np.delete(s, inds)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Purpose
----
Read physio data from a CMRR Multiband generated DICOM file (DICOM spectroscopy format)
Current CMRR MB sequence version : VE11C R016a

Usage
----
dcm2physio.py -i <CMRR DICOM Physio>

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2018-03-29 JMT From scratch
2018-11-19 JMT Port parsing logic from extractCMRRPhysio.m (Ed Auerbach)

References
----
Matlab parser: https://github.com/CMRR-C2P/MB/blob/master/extractCMRRPhysio.m

License
----
MIT License

Copyright (c) 2017-2018 Mike Tyszka

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

__version__ = '1.1.1'


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

    if (n_points % n_rows):
        print('* Points (%d) is not an integer multiple of rows (%d) - exiting' % (n_points, n_rows))
        sys.exit(-1)

    n_cols = int(n_points / n_rows)

    if (n_points % 1024):
        print('* Columns (%d) is not an integer multiple of 1024 (%d) - exiting' % (n_cols, 1024))
        sys.exit(-1)

    n_waves = int(n_cols / 1024)
    wave_len = int(n_points / n_waves)

    for wc in range(n_waves):

        print('')
        print('Parsing waveform %d' % wc)

        offset = wc * wave_len

        wave_data = physio_data[slice(offset, offset+wave_len)]

        data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
        fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
        fname = wave_data[slice(8, 8+fname_len)]

        print('Data length     : %d' % data_len)
        print('Filename length : %d' % fname_len)
        print('Filename        : %s' % fname)

        # Extract waveform log byte data
        log_bytes = wave_data[slice(1024, 1024+data_len)]

        # Parse waveform log
        t_puls, s_puls, t_resp, s_resp = parse_log(log_bytes)
#
#     # Zero time origin (identical offset for all waveforms) and scale to seconds
#     t_puls = (t_puls - t_puls[0]) * dt_puls
#     t_resp = (t_resp - t_resp[0]) * dt_resp
#
#     # Resample respiration waveform to match pulse waveform timing
#     f = interp1d(t_resp, s_resp, kind='cubic', fill_value='extrapolate')
#     s_resp_i = f(t_puls)
#
#     # Create a dataframe from a data dictionary
#     d = {'Time_s':t_puls, 'Pulse':s_puls, 'Resp': s_resp_i}
#     df = pd.DataFrame(d)
#
#     # Export pulse and respiratory waveforms to TSV file
#     tsv_fname = os.path.splitext(physio_dcm)[0]+'.tsv'
#     print('Saving pulse and respiratory waveforms to %s' % tsv_fname)
#     df.to_csv(tsv_fname,
#               sep='\t',
#               columns=['Time_s', 'Pulse', 'Resp'],
#               index=False,
#               float_format='%0.3f')
#
def parse_log(log_bytes):

    # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
    physio_string = log_bytes.decode('utf-8', 'ignore')

    # CMRR DICOM physio log has \n separated lines
    physio_lines = physio_string.splitlines()

    # Parse the pulse and respiratory waveforms to a numpy array

    t_puls = []
    t_resp = []
    s_puls = []
    s_resp = []

    current_waveform = ''

    for line in physio_lines:

        parts = line.split()

        if len(parts) > 2:

            if 'ScanDate' in parts[0]:
                scan_date = parts[2]

            if 'LogDataType' in parts[0]:
                current_waveform = parts[2]

            if 'SampleTime' in parts[0]:
                dt_puls = float(parts[2]) * 1e-3

                if 'RESP' in current_waveform:
                    dt_resp = float(parts[2]) * 1e-3

            if 'PULS' in parts[1]:
                t_puls.append(float(parts[0]))
                s_puls.append(float(parts[2]))

            if 'RESP' in parts[1]:
                t_resp.append(float(parts[0]))
                s_resp.append(float(parts[2]))

    return t_puls, s_puls, t_resp, s_resp


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()


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
import argparse
import numpy as np
import pydicom
from scipy.interpolate import interp1d


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DICOM files to BIDS-compliant Nifty structure')

    parser.add_argument('-i', '--indir', default='dicom',
                        help='DICOM input directory with Subject/Session/Image organization [dicom]')

    parser.add_argument('-o', '--outdir', default='source',
                        help='Output BIDS source directory [source]')

    parser.add_argument('--no-sessions', action='store_true', default=False,
                        help='Do not use session sub-directories')

    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')

    # Parse command line arguments
    args = parser.parse_args()
    dcm_root_dir = os.path.realpath(args.indir)
    no_sessions = args.no_sessions
    overwrite = args.overwrite

    # Create DICOM object
    d = pydicom.read_file('physio.dcm')

    # Extract data from Siemens spectroscopy tag (0x7fe1, 0x1010)
    # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
    physio_string = d[0x7fe1, 0x1010].value.decode('utf-8', 'ignore')

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

                if 'PULS' in current_waveform:
                    dt_puls = float(parts[2]) * 1e-3

                if 'RESP' in current_waveform:
                    dt_resp = float(parts[2]) * 1e-3

            if 'PULS' in parts[1]:
                t_puls.append(float(parts[0]))
                s_puls.append(float(parts[2]))

            if 'RESP' in parts[1]:
                t_resp.append(float(parts[0]))
                s_resp.append(float(parts[2]))

    # Convert to numpy arrays
    t_puls = np.array(t_puls)
    t_resp = np.array(t_resp)
    s_puls = np.array(s_puls)
    s_resp = np.array(s_resp)

    # Zero time origin (identical offset for all waveforms) and scale to seconds
    t_puls = (t_puls - t_puls[0]) * dt_puls
    t_resp = (t_resp - t_resp[0]) * dt_resp

    # Resample respiration waveform to match pulse waveform timing
    f = interp1d(t_resp, s_resp, kind='cubic', fill_value='extrapolate')
    s_resp_i = f(t_puls)

    # Export pulse and respiratory waveforms to TSV file


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
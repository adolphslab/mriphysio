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

import os
import argparse

from .physiodicom import PhysioDicom
from .ecg import HeartRate


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DICOM Physiolog to BIDS-compliant TSV')
    parser.add_argument('-i', '--infile', required=True, help='CMRR physio DICOM file')
    parser.add_argument('-o', '--outfile', default='', help='CMRR physio DICOM file')
    args = parser.parse_args()

    physio_dcm = args.infile
    if len(args.outfile) > 0:
        physio_tsv = args.outfile
    else:
        physio_tsv = f"{os.path.splitext(physio_dcm)[0]}.tsv"

    # Load, convert and save DICOM physio log
    physio = PhysioDicom(physio_dcm)
    waveforms = physio.convert()

    # Save TSV file of all waveforms
    physio.save(physio_tsv)

    # Save plot of first few seconds of all waveforms to a PNG image
    preview_png = physio_tsv.replace('.tsv', '_preview.png')
    physio.save_preview(preview_png)

    # Optional heart rate output if ECG present
    if physio.have_ecg:

        # Heart rate BIDS TSV from physio TSV
        hr_tsv = physio_tsv.replace('.tsv', '_heartrate.tsv')

        heart_rate = HeartRate(waveforms['Time_s'].values, waveforms[['ECG1', 'ECG2', 'ECG3', 'ECG4']].values)
        heart_rate.estimate()
        heart_rate.save(hr_tsv)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()


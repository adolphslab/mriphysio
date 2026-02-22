#!/usr/bin/env python
"""
Purpose
----
Estimate the respiration waveform from the phase difference of a complex EPI time series.

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

License
----
MIT License

Copyright (c) 2024 Mike Tyszka

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
import os.path as op
import argparse

from .phaseresp import PhaseRespWave


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Estimate respiratory waveform from complex EPI and save as BIDS-compliant TSV')
    parser.add_argument('-p', '--phasediff', required=True, help='Nifti 3D x time phase difference timeseries')
    parser.add_argument('-w', '--weightmask', required=True, help='Nifti weighting mask image')
    parser.add_argument('-m', '--method', default='weighted_mean', help='Spatially weighted mean method')
    parser.add_argument('-n', '--npca', default=1, type=int, help='Number of PCA components to retain')  
    parser.add_argument('-o', '--outfile', required=True, help='BIDS format repiratory waveform TSV')
    args = parser.parse_args()

    resp_tsv = args.outfile
    out_dir = op.dirname(resp_tsv)
    bids_stub = op.basename.splitext(resp_tsv)[0]

    # Estimate the respiratory waveform
    resp_obj = PhaseRespWave(args.phasediff, args.weightmask, args.method, args.npca)
    resp_obj.estimate()
    resp_obj.to_bids(out_dir, bids_stub)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()


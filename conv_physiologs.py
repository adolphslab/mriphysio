#!/usr/bin/env python
"""
Convert Siemens physio DICOM files to BIDS tsv files and populate to correct subject/session folders directly

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2021-08-02 JMT From scratch

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
import sys
from glob import glob
import bids
import pydicom
import json
import numpy as np
from datetime import datetime as dt
from scipy.interpolate import interp1d


def date2version(datestr, study_desc):
    """
    Map session date to a Conte Core protocol versions.
    Note that SocialBattery sessions from 2017 are dealt with separately from
    the versioned Conte Core protocols.
    """

    date_obj  = dt.strptime(os.path.basename(datestr),'%Y%m%d')

    # Handle special cases first then infer version from scan date
    if "Social_Battery" in study_desc:
        ver = 'socbat'
    elif date_obj > dt(2018,4,1):
        ver = '2p2'
    elif date_obj > dt(2015,5,10):
        ver = '1p3p1'
    elif date_obj > dt(2015,1,20):
        ver = '1p3p0'
    elif date_obj > dt(2013,8,26):
        ver = '1p2'
    elif date_obj > dt(2012,10,20):
        ver = '1p1'
    else:
        ver = 'unknown'

    return ver


def convert_physio(ds, dest_dir, physio_stub, verbose=False):
    """
    Purpose
    ----
    Read physio data from a CMRR Multiband generated DICOM file (DICOM spectroscopy format)
    Current CMRR MB sequence version : VE11C R016a
    Adapt parsing logic from extractCMRRPhysio.m. Also see adolphslab/mriphysio/dcm2physio.py

    Author:
    ----
    Mike Tyszka, Caltech

    References
    ----
    Parser: Adapted from https://github.com/CMRR-C2P/MB/blob/master/extractCMRRPhysio.m

    :param ds: pydicom data structure
    :param dest_dir: str
        Destination BIDS func directory for this subject/session
    :param physio_stub: str
        Filename stub for json and tsv physiolog files
    :return:
    """

    # Extract data from Siemens spectroscopy tag (0x7fe1, 0x1010)
    # Yields one long byte array
    physio_data = ds[0x7fe1, 0x1010].value

    # Extract relevant info from header
    n_points = len(physio_data)
    n_rows = ds.AcquisitionNumber

    if (n_points % n_rows):
        print('* Points (%d) is not an integer multiple of rows (%d) - exiting' % (n_points, n_rows))
        sys.exit(-1)

    n_cols = int(n_points / n_rows)

    if (n_points % 1024):
        print('* Columns (%d) is not an integer multiple of 1024 (%d) - exiting' % (n_cols, 1024))
        sys.exit(-1)

    n_waves = int(n_cols / 1024)
    wave_len = int(n_points / n_waves)

    # Init time and waveform vectors
    t_puls, s_puls, dt_puls = [], [], 1.0
    t_resp, s_resp, dt_resp = [], [], 1.0

    for wc in range(n_waves):

        if verbose:
            print('')
            print('Parsing waveform %d' % wc)

        offset = wc * wave_len

        wave_data = physio_data[slice(offset, offset+wave_len)]

        data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
        fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
        fname = wave_data[slice(8, 8+fname_len)]

        if verbose:
            print('Filename        : %s' % fname)
            print('Filename length : %d' % fname_len)
            print('Data length     : %d' % data_len)

        # Extract waveform log byte data
        log_bytes = wave_data[slice(1024, 1024+data_len)]

        # Parse waveform log
        waveform_name, t, s, dt = parse_log(log_bytes)

        if "PULS" in waveform_name:
            t_puls, s_puls, dt_puls = t, s, dt

        if "RESP" in waveform_name:
            t_resp, s_resp, dt_resp = t, s, dt

    # Zero time origin (identical offset for all waveforms) and scale to seconds
    t_puls = (t_puls - t_puls[0]) * dt_puls
    t_resp = (t_resp - t_resp[0]) * dt_resp

    # Resample respiration waveform to match pulse waveform timing
    f = interp1d(t_resp, s_resp, kind='cubic', fill_value='extrapolate')
    s_resp_i = f(t_puls)

    # Create a two-column numpy array and write to a TSV file
    out_array = np.array([s_puls, s_resp_i]).T

    tsv_fname = os.path.join(dest_dir, physio_stub + '.tsv')
    print(f'  Saving pulse and resp waveforms to {tsv_fname}')
    np.savetxt(tsv_fname, out_array, fmt='%0.3f', delimiter='\t')

    # Create a metadata dictionary and write to a JSON file
    meta_dict = {
        'SamplingFrequency': 1.0/dt_puls,
        'StartTime': 0.0,
        'Columns': ['Pulse', 'Respiration']
    }

    json_fname = os.path.join(dest_dir, physio_stub + '.json')
    print(f'  Saving waveform metadata to {json_fname}')
    with open(json_fname, 'w') as fd:
        json.dump(meta_dict, fd)


def parse_log(log_bytes, verbose=False):

    # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors
    physio_string = log_bytes.decode('utf-8', 'ignore')

    # CMRR DICOM physio log has \n separated lines
    physio_lines = physio_string.splitlines()

    # Init parameters and lists
    uuid = "UNKNOWN"
    waveform_name = "UNKNOWN"
    scan_date = "UNKNOWN"
    dt = 1.0
    t_list, s_list = [], []

    for line in physio_lines:

        # Divide the line at whitespace
        parts = line.split()

        # Data lines have the form "<tag> = <value>" or "<time> <name> <signal>"
        if len(parts) == 3:

            p1, p2, p3 = parts

            if 'UUID' in p1:
                uuid = p3

            if 'ScanDate' in p1:
                scan_date = p3

            if 'LogDataType' in p1:
                waveform_name = p3

            if 'SampleTime' in p1:
                dt = float(p3) * 1e-3

            if 'PULS' in p2 or 'RESP' in p2:
                t_list.append(float(p1))
                s_list.append(float(p3))

    if verbose:
        print(f'UUID            : {uuid}')
        print(f'Scan date       : {scan_date}')
        print(f'Waveform type   : {waveform_name}')
        print(f'Sample Time     : {dt * 1000.0} ms')

    # Return numpy arrays
    return waveform_name, np.array(t_list), np.array(s_list), dt


def main():

    # bids_dir = '/data2/conte'
    bids_dir = '/Users/jmt/Data/Physiologs'
    physio_src_dir = os.path.join(bids_dir, 'sourcedata', 'physio')

    physio_list = sorted(glob(os.path.join(physio_src_dir, '*.dcm')))
    n_physios = len(physio_list)

    if n_physios > 0:

        print(f'Found {n_physios} physiolog DICOMs in {physio_src_dir}')

        for dcm_fname in physio_list:

            print('')
            print(f"Processing {os.path.basename(dcm_fname)}")

            # Parse filename
            keys = bids.layout.parse_file_entities(dcm_fname)

            sub_id = f"sub-{keys['subject']}"
            task_id = f"task-{keys['task']}"

            # Load DICOM physiolog
            ds = pydicom.read_file(dcm_fname)

            try:
                study_desc = str(ds.StudyDescription)
            except AttributeError:
                study_desc = 'Unknown'

            # Convert physiolog series time ('HHMMSS.FFFFFF') to datetime
            physio_acqtime = dt.strptime(ds.SeriesTime, '%H%M%S.%f')
            print(f"  Physiolog acquisition Time : {dt.strftime(physio_acqtime, '%H:%M:%S.%f')}")

            # Convert session ID from date to protocol version
            core_ver = date2version(str(ds.StudyDate), study_desc)
            ses_id_ver = 'ses-core{}'.format(core_ver)

            dest_dir = os.path.join(bids_dir, sub_id, ses_id_ver, 'func')

            print(f'  Destination directory {dest_dir}')
            if os.path.isdir(dest_dir):

                print(f'  Searching for BOLD JSON files for {task_id}')

                # Predict associated _bold JSON file for this sub/ses/task
                # There may be more than one run, in which case find the bold recon immediate before the physiolog
                bold_list = sorted(glob(os.path.join(dest_dir, f'*{task_id}*_bold.json')))
                n_bold_json = len(bold_list)

                if n_bold_json > 0 :

                    # Physiologs are reconstructed at the end of a BOLD series
                    # and should have a positive time delta. Look for the smallest positive
                    # time delta between the physiolog and BOLD series
                    min_pos_delta = 1e31
                    closest_bold = ''

                    for bold_json_fname in bold_list:

                        with open(bold_json_fname, 'r') as fd:
                            json_dict = json.load(fd)

                        # Get the acquisition time for the BOLD series
                        # Remove colons and convert to float for comparison to physiolog time
                        bold_acqtime = dt.strptime(json_dict['AcquisitionTime'], '%H:%M:%S.%f')
                        print(f"    BOLD acquisition time : {dt.strftime(bold_acqtime, '%H:%M:%S.%f')}")

                        time_delta = (physio_acqtime - bold_acqtime).total_seconds()
                        print(f"    Time difference : {time_delta} s")

                        if (time_delta > 0.0) & (time_delta < min_pos_delta):
                            min_pos_delta = time_delta
                            closest_bold = bold_json_fname

                    print(f'  Closest BOLD JSON : {closest_bold}')

                    # Construct BIDS physio output stub (without extension)
                    physio_stub = closest_bold.replace('_bold.json', '_physio')

                    # Convert physiolog and write to correct destination func folder
                    convert_physio(ds, dest_dir, physio_stub)

                else:

                    print(f'* No BOLD JSON files found for {task_id} - skipping')

            else:

                print('* Destination directory not found - skipping')

    else:

        print(f'* No physiolog DICOMs found in {physio_src_dir}')

if '__main__' in __name__:

    bids.config.set_option('extension_initial_dot', True)

    main()

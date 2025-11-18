"""
BIDS physiological file output function

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center
DATES  : 2024-12-13 JMT Consolidate BIDS physio handling here
"""

import json
import os
import os.path as op


def save_bids_physio(bids_outdir, bids_stub, physio_df):
    """

    :param bids_outdir: str, Path-like
        Directory where BIDS files will be written
    :param bids_stub: str
        BIDS-compatible stub for filename, including recording-[TYPE]_physio suffix
    :param physio_df: pandas DataFrame
        Physiological recording data including Time column
    :return:
    """

    """
    From BIDS v1.10.0:
    - Physiological recordings such as cardiac and respiratory signals MAY be specified using a compressed
    tabular file (TSV.GZ file) and a corresponding JSON file for storing metadata fields.
    - In the template filenames, the <matches> part corresponds to task filename before the suffix.
    - <matches>_physio.tsv.gz files MUST NOT include a header line, as established by the common-principles.
    As a result, when supplying a <matches>_physio.tsv.gz file, an accompanying <matches>_physio.json MUST be
    present to indicate the column names.
    - All columns require a specification in the JSON sidecar. Column headings only will generate a warning at validation

    :param out_dir: str, Path-like
        Directory where BIDS files will be written to.
    :param bids_stub: str
    :param physio_df: pandas DataFrame
        physiological timestamp and waveform data
    :param json_dict: dict
        BIDS JSON sidecar metadata

    :return:
    """

    # Output physio TSV and JSON filenames
    tsv_fname = op.join(bids_outdir, f'{bids_stub}.tsv')
    json_fname = op.join(bids_outdir, f'{bids_stub}.json')

    # Save respiration waveform to BIDS physio TSV
    # Compressed TSV physio files should NOT have header row
    print(f'Saving respiratory waveform to {tsv_fname}')
    physio_df.to_csv(
        tsv_fname,
        sep='\t',
        index=False,
        header=False,
        float_format='%0.6f'
    )

    # gzip the TSV file
    tsv_gz_fname = tsv_fname + '.gz'
    print(f'Compressing TSV to {tsv_gz_fname}')
    os.system(f'gzip -f {tsv_fname}')

    # Populate a dictionary for the JSON sidecar

    # Infer sampling frequency from time column
    dt = physio_df['Time'].diff().mean()
    sf = 1.0 / dt

    json_dict = {
        "SamplingFrequency": sf,
        "StartTime": physio_df['Time'].min(),
        "Columns": physio_df.columns.tolist(),
        "Time": {
            "Description": "Sample timestamps",
            "Units": "s"
        }
    }

    # Full descriptions for all possible columns
    if 'ECG1' in physio_df.columns:
        json_dict['ECG1'] = {
            "Description": "Electrocardiogram channel 1",
            "Units": "arbitrary"
        }
    if 'ECG2' in physio_df.columns:
        json_dict['ECG2'] = {
            "Description": "Electrocardiogram channel 2",
            "Units": "arbitrary"
        }
    if 'ECG3' in physio_df.columns:
        json_dict['ECG3'] = {
            "Description": "Electrocardiogram channel 3",
            "Units": "arbitrary"
        }
    if 'ECG4' in physio_df.columns:
        json_dict['ECG4'] = {
            "Description": "Electrocardiogram channel 4",
            "Units": "arbitrary"
        }
    if 'RESP' in physio_df.columns:
        json_dict['RESP'] = {
            "Description": "Respiratory waveform",
            "Units": "arbitrary"
        }
    if 'ACQ' in physio_df.columns:
        json_dict['ACQ'] = {
            "Description": "Acquisition trigger",
            "Units": "arbitrary"
        }

    # Finally write JSON sidecar
    print(f'Saving JSON sidecar to {json_fname}')
    write_json(json_fname, json_dict, overwrite=True)

def write_json(json_fname, json_dict, overwrite=False):

    if op.isfile(json_fname):
        if overwrite:
            create_file = True
        else:
            create_file = False
    else:
        create_file = True

    if create_file:
        with open(json_fname, 'w') as fd:
            json.dump(json_dict, fd, indent=4, separators=(',', ':'))

def make_bids_stub(pname):
    """
    Generate a BIDS-compatible stub from a file path
    
    :param pname: str
        Full path to file
    """

    # Strip all extensions from basename
    base = op.basename(pname).split('.')[0]
    
    # Remove the final underscore-delimited string
    parts = base.split('_')
    if len(parts) > 1:
        bstub = '_'.join(parts[:-1])
    else:
        bstub = base
    
    return bstub
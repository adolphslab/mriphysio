# mriphysio
Utility classes and functions for handling the following:
- Respiration waveform reconstruction from complex-valued or phase-only EPI timeseries
- Siemens/CMRR MRI physiological DICOM
- Siemens MPCU logs


## Respiratory waveform from complex or phase-only EPI timeseries
The installed python package provides a command line function `epi2resp` that generates a BIDS-format respiratory TSV file from a complex-valued EPI timeseries.

### Usage
usage: epi2resp [-h] -p PHASEDIFF -w WEIGHTMASK [-m METHOD] [-n NPCA] -o OUTFILE

Estimate respiratory waveform from complex EPI and save as BIDS-compliant TSV

options:
  -h, --help            show this help message and exit
  -p PHASEDIFF, --phasediff PHASEDIFF
                        Nifti 3D x time phase difference timeseries
  -w WEIGHTMASK, --weightmask WEIGHTMASK
                        Nifti weighting mask image
  -m METHOD, --method METHOD
                        Spatially weighted mean method
  -n NPCA, --npca NPCA  Number of PCA components to retain
  -o OUTFILE, --outfile OUTFILE
                        BIDS format repiratory waveform TSV

### PhaseRespWave Class
The `PhaseRespWave` class is defined in `mriphysio/phaseresp.py` and `mriphysio/epi2resp.py` shows example use of the class for pipeline integration.

## Notes on Siemens physiology log parsing

### VE11C CMRR Multiband DICOM Physiological Logs
R016a of the Prisma version of CMRR multiband EPI sequences supports output of physiological logging as a DICOM file.
The waveforms are encoded in the Siemens Spectroscopy tag (0x7fe1, 0x1010) and match the acquisition period of the series accurately. A jupyter notebook demonstrating handling of an example physio DICOM is provided in `data/notebooks/`

### VB17A MPCU Matlab Functions
Legacy MPCU text log handling functions for Matlab (mpcu*.m) are provided in `matlab/`

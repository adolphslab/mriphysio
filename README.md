# mriphysio
Utility classes and functions for handling the following:
- Siemens/CMRR MRI physiological DICOM
- Siemens MPCU logs
- Respiration waveform reconstruction from EPI phase timeseries

## VE11C CMRR Multiband DICOM Physiological Logs
R016a of the Prisma version of CMRR multiband EPI sequences supports output of physiological logging as a DICOM file.
The waveforms are encoded in the Siemens Spectroscopy tag (0x7fe1, 0x1010) and match the acquisition period of the series accurately. A jupyter notebook demonstrating handling of an example physio DICOM is provided.

## VB17A MPCU Matlab Functions
Legacy MPCU text log handling functions for Matlab (mpcu*.m)

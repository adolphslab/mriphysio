# mriphysio
MRI physiological log handling functions

## CMRR Multiband DICOM Physiological Logs
R016a of the Prisma version (VE11C) of CMRR multiband EPI sequences supports output of physiological logging as a DICOM file.
This uses the Siemens Spectroscopy tag (0x7fe1, 0x1010) to encoding all physio waveforms acquired during the
acquired EPI volumes. A jupyter notebook demonstrating handling of an example physio DICOM are provided.

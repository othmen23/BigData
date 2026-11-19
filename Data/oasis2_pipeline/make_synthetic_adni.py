"""FAKE ADNI-style DICOM folders (same layout as your T1w_Imaging_Cohort_MRI_1) to test the tools."""
import sys
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

root = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_adni")
rng = np.random.default_rng(0)
for k in range(12):
    ptid = f"{int(rng.integers(1, 999)):03d}_S_{1000 + k:04d}"
    for date in sorted(rng.choice(["2006-06-20", "2007-06-25", "2008-07-01"], 3, replace=False)):
        folder = root / "ADNI" / ptid / "MP-RAGE" / f"{date}_10_37_15.0" / f"I{17000 + int(rng.integers(0, 9999))}"
        folder.mkdir(parents=True, exist_ok=True)
        for s in range(30):
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
            ds.SOPClassUID, ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPClassUID, ds.file_meta.MediaStorageSOPInstanceUID
            ds.Rows = ds.Columns = 64
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
            ds.ImagePositionPatient = [-60 + 1.2 * s, -128, 128]
            ds.InstanceNumber = 30 - s                      # shuffled on purpose: reader must sort by position
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
            ds.PixelData = rng.integers(0, 800, (64, 64)).astype(np.uint16).tobytes()
            ds.save_as(folder / f"ADNI_{ptid}_MR_MP-RAGE__br_raw_{s:03d}.dcm", enforce_file_format=True)
print("synthetic ADNI written to", root)

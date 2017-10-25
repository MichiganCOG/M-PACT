
## TODO Finish implementation of generalized hdf5 file generation

import h5py
import cv2
import numpy as np
import os


""" Function to generate HDF5 files independently """
def genHDF5(baseDataPath, videos, baseDestPath, size=224, fName='trainlist', idx=0):
    # Create HDF5 file
    fHDF5 = h5py.File(baseDestPath + fName+'_['+str(idx)+']_.hdf5','w')


    # Open data files



    # Load videos


    # Group them???
    curGrp = fHDF5.create_group()

    # Convert to hdf5 frame by frame
    for frame in video:
        data = np.concatenate(data, frame)

    curGrp['Label'] = label
    curGrp['Data'] = data

""" Generating HDF5 files based on text file input """

import h5py
import cv2
import numpy as np
import os
import argparse
import multiprocessing
import time
import subprocess

""" Function to generate HDF5 files independently """
def genHDF5(baseDataPath, videos, baseDestPath, size=224, fName='trainlist', idx=0):

    # Create HDF5 file
    fHDF5 = h5py.File(baseDestPath + fName+'_['+str(idx)+']_.hdf5','w')
        
    # Loop over all videos in this chunk
    for line in videos:
        
        # Create video subgroup
        action_class    = os.path.split(line)[0]
        video_name, ext = os.path.splitext(os.path.split(line)[1].split(' ')[0])
        ext             = ext.replace('\r','').replace('\n','')
        cur_grp         = fHDF5.create_group(video_name)
        video_path      = os.path.join(action_class, video_name + ext)

        assert(os.path.isfile(os.path.join(baseDataPath, video_path)))

        # Load original video
        data1  = []
        data2  = []
        video  = cv2.VideoCapture(os.path.join(baseDataPath, video_path))
        count  = 0

        flag, frame = video.read()

        while flag: 
            
            H,W,C = frame.shape

            if count < 150:
                if count == 0:
                    data1 = frame.reshape(1, H, W, C)
                else:
                    data1 = np.concatenate((data1, frame.reshape(1, H, W, C)))
            else:
                if count == 150:
                    data2 = frame.reshape(1, H, W, C)
                else:
                    data2 = np.concatenate((data2, frame.reshape(1, H, W, C)))

            count += 1
            flag, frame= video.read()

        if not len(data2) == 0:
            data = np.concatenate((data1, data2))
        else:
            data = data1

        assert(len(data.shape)>0)

        if len(os.path.split(line)[1].split(' ')) > 1:
            label = int(os.path.split(line)[1].split(' ')[1]) 
        else:
            label = -1
            
        # Assign values Data and Label sub groups of the video
        cur_grp["Label"] = label
        cur_grp["Data"]  = data

       
    fHDF5.close()
 

""" Function to generate HDF5 files in chunks """
def genData(vidsFile, baseDataPath, baseDestPath, size=224, chunk=1000, fName='trainlist'):
    
    # COMPLETE FLAG
    cFLAG = False

    # Read input file
    fin   = open(vidsFile, 'r')
    lines = fin.readlines()
    jobs = []

    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(30)

    for idxIter in np.arange(0, len(lines), chunk*30):
        
        if idxIter <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter:(idxIter+ chunk)], baseDestPath, size, fName, idxIter/chunk))

        if idxIter+chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+1*chunk:((idxIter+1*chunk) + chunk)], baseDestPath, size, fName, (idxIter+chunk)/chunk))

        if idxIter+2*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+2*chunk:((idxIter+2*chunk) + chunk)], baseDestPath, size, fName, (idxIter+2*chunk)/chunk))

        if idxIter+3*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+3*chunk:((idxIter+3*chunk) + chunk)], baseDestPath, size, fName, (idxIter+3*chunk)/chunk))

        if idxIter+4*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+4*chunk:((idxIter+4*chunk) + chunk)], baseDestPath, size, fName, (idxIter+4*chunk)/chunk))

        if idxIter+5*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+5*chunk:((idxIter+5*chunk) + chunk)], baseDestPath, size, fName, (idxIter+5*chunk)/chunk))

        if idxIter+6*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+6*chunk:((idxIter+6*chunk) + chunk)], baseDestPath, size, fName, (idxIter+6*chunk)/chunk))

        if idxIter+7*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+7*chunk:((idxIter+7*chunk) + chunk)], baseDestPath, size, fName, (idxIter+7*chunk)/chunk))

        if idxIter+8*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+8*chunk:((idxIter+8*chunk) + chunk)], baseDestPath, size, fName, (idxIter+8*chunk)/chunk))

        if idxIter+9*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+9*chunk:((idxIter+9*chunk) + chunk)], baseDestPath, size, fName, (idxIter+9*chunk)/chunk))

        if idxIter+10*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+10*chunk:((idxIter+10*chunk) + chunk)], baseDestPath, size, fName, (idxIter+10*chunk)/chunk))

        if idxIter+11*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+11*chunk:((idxIter+11*chunk) + chunk)], baseDestPath, size, fName, (idxIter+11*chunk)/chunk))

        if idxIter+12*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+12*chunk:((idxIter+12*chunk) + chunk)], baseDestPath, size, fName, (idxIter+12*chunk)/chunk))

        if idxIter+13*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+13*chunk:((idxIter+13*chunk) + chunk)], baseDestPath, size, fName, (idxIter+13*chunk)/chunk))

        if idxIter+14*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+14*chunk:((idxIter+14*chunk) + chunk)], baseDestPath, size, fName, (idxIter+14*chunk)/chunk))

        if idxIter+15*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+15*chunk:((idxIter+15*chunk) + chunk)], baseDestPath, size, fName, (idxIter+15*chunk)/chunk))

        if idxIter+16*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+16*chunk:((idxIter+16*chunk) + chunk)], baseDestPath, size, fName, (idxIter+16*chunk)/chunk))

        if idxIter+17*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+17*chunk:((idxIter+17*chunk) + chunk)], baseDestPath, size, fName, (idxIter+17*chunk)/chunk))

        if idxIter+18*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+18*chunk:((idxIter+18*chunk) + chunk)], baseDestPath, size, fName, (idxIter+18*chunk)/chunk))

        if idxIter+19*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+19*chunk:((idxIter+19*chunk) + chunk)], baseDestPath, size, fName, (idxIter+19*chunk)/chunk))

        if idxIter+20*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+20*chunk:((idxIter+20*chunk) + chunk)], baseDestPath, size, fName, (idxIter+20*chunk)/chunk))

        if idxIter+21*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+21*chunk:((idxIter+21*chunk) + chunk)], baseDestPath, size, fName, (idxIter+21*chunk)/chunk))

        if idxIter+22*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+22*chunk:((idxIter+22*chunk) + chunk)], baseDestPath, size, fName, (idxIter+22*chunk)/chunk))

        if idxIter+23*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+23*chunk:((idxIter+23*chunk) + chunk)], baseDestPath, size, fName, (idxIter+23*chunk)/chunk))

        if idxIter+24*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+24*chunk:((idxIter+24*chunk) + chunk)], baseDestPath, size, fName, (idxIter+24*chunk)/chunk))

        if idxIter+25*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+25*chunk:((idxIter+25*chunk) + chunk)], baseDestPath, size, fName, (idxIter+25*chunk)/chunk))

        if idxIter+26*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+26*chunk:((idxIter+26*chunk) + chunk)], baseDestPath, size, fName, (idxIter+26*chunk)/chunk))

        if idxIter+27*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+27*chunk:((idxIter+27*chunk) + chunk)], baseDestPath, size, fName, (idxIter+27*chunk)/chunk))

        if idxIter+28*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+28*chunk:((idxIter+28*chunk) + chunk)], baseDestPath, size, fName, (idxIter+28*chunk)/chunk))

        if idxIter+29*chunk <= len(lines)-1:
            pool.apply_async(genHDF5, (baseDataPath, lines[idxIter+29*chunk:((idxIter+29*chunk) + chunk)], baseDestPath, size, fName, (idxIter+29*chunk)/chunk))


        pool.close()
        pool.join()
        pool = multiprocessing.Pool(30)

    # Final exit case
    pool.close()
    pool.join()


   
if __name__=="__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidsFile', action= 'store', required= True,
            help= 'Input information file to process and from which to generate data')
    parser.add_argument('--baseDataPath', action= 'store', required= True,
            help= 'Original dataset directory')
    parser.add_argument('--baseDestPath', action= 'store', required= True,
            help= 'Destination directory')
    parser.add_argument('--chunk', action= 'store', required= True, type= int,
            help= 'Number of videos to aggregate together into a single HDF5 file')
    parser.add_argument('--fName', action= 'store', required= True,
            help= 'Name of HDF5 file, prefix')

    args = parser.parse_args()

    genData(vidsFile= args.vidsFile, 
            baseDataPath= args.baseDataPath, 
            baseDestPath= args.baseDestPath, 
            chunk= args.chunk,
            fName= args.fName) 

import h5py
import numpy as np
import os

def load_dataset(model, index, fName, baseDataPath, vidFile, classIndFile, size, isTraining, dataset, chunk=100):

    fin = open(vidFile,'r+')
    lines = fin.readlines()


    data = []
    label = []

    # 1. Load HDF5 file
    fHDF5 = h5py.File(os.path.join(baseDataPath, fName)+'_['+str(int(index/chunk))+']_.hdf5', 'r')

    data =fHDF5[os.path.split(lines[index].split('.avi')[0])[1]]['Data'].value
#    data =fHDF5[os.path.split(ps.path.splitext(lines[index])[0])[1]]['Data'].value
    label = fHDF5[os.path.split(lines[index].split('.avi')[0])[1]]['Label'].value
    #label = fHDF5[os.path.split(os.path.splitext(lines[index])[0])[1]]['Label'].value
    if (dataset=='UCF101' or dataset =='UCF101Rate') and label != -1:
        label = label-1

    if label==-1:
        # Fail safe to obtain indices if test data with no labels is provided
        CLASS = os.path.split(lines[index].split(' ')[0])[0]
        fIn = open(classIndFile,'r')
        classLines = fIn.readlines()
        classDict = {}
        for line in classLines:
            line1 = line.split(" ")[0]
            line2 = line.split("\n")[0].split(" ")[1]
            classDict[line2] = int(line1)-1
        label = classDict[CLASS]

    if dataset =='HMDB51' or dataset=="HMDB51Rate":
        data = np.delete(data,np.where(np.arange(1,data.shape[0])%6==0)[0].astype('int32'), axis=0)

    vid_clips = model.preprocess(index, data, label, size, isTraining)


    fHDF5.close()

    return vid_clips, label

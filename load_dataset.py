""" THIS FILE IS USED TO LOAD THE VIDEO DATA FROM HDF5 FILES """

import h5py
import numpy as np
import os

def load_dataset(model, index, f_name, base_data_path, vid_file, class_ind_file, size, is_training, dataset, chunk=100):
    """ 
    Args:
        :model:          Model definition from tf-acitivity-recognition framework
        :index:          Integer indicating the index of video from a text file
        :f_name:         Prefix of HDF5 files to be loaded
        :base_data_path: Root directory of datasets
        :vid_file:       Full path to file containing list of videos to be loaded
        :class_ind_file: Full path to class index file generated for a dataset
        :size:           List detailing height and width of final frames
        :is_training:    Boolean variable to indicate current phase
        :dataset:        Name of dataset

    Return:
        :vid_clips: Final video clips
        :labels:    Label of selected video clips

    """

    fin   = open(vid_file,'r+')
    lines = fin.readlines()

    data  = []
    label = []

    # 1. Load HDF5 file
    fHDF5 = h5py.File(os.path.join(base_data_path, f_name)+'_['+str(int(index/chunk))+']_.hdf5', 'r')

    # 2. Grab relevant data from HDF5 file
    data  = fHDF5[os.path.split(os.path.splitext(lines[index])[0])[1]]['Data'].value
    label = fHDF5[os.path.split(os.path.splitext(lines[index])[0])[1]]['Label'].value

    fHDF5.close()

    # 3. Reduce label by 1 if UCF101-based datasets are loaded (LEGACY LOADING ISSUE)
    if ('UCF101' in dataset) and label != -1:
        label = label-1

    # END IF

    # 4. If loaded video is from testing set with no valid label
    if label==-1:
        # Fail safe to obtain indices if test data with no labels is provided
        Class       = os.path.split(lines[index].split(' ')[0])[0]
        f_class_in  = open(class_ind_file,'r')
        class_lines = f_class_in.readlines()
        class_dict  = {}

        for line in class_lines:
            line1 = line.split(" ")[0]
            line2 = line.split("\n")[0].split(" ")[1]
            class_dict[line2] = int(line1)-1

        # END FOR

        label = class_dict[Class]

    # END IF

    # 5. Resample to 25fps capture if loaded dataset is HMDB51-based
    if 'HMDB51' in dataset:
        data = np.delete(data,np.where(np.arange(1,data.shape[0])%6==0)[0].astype('int32'), axis=0)

    # END IF

    # 6. Preprocess data based on the relevant model
    vid_clips = model.preprocess(index, data, label, size, is_training)

    return vid_clips, label

import os

import numpy      as np
import tensorflow as tf


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_checkpoint(model, dataset, experiment_name):
    try:
        checkpoints_file = os.path.join('results', model, dataset, experiment_name, 'checkpoints', 'checkpoint')
        f = open(checkpoints_file, 'r')
        filename = f.readline()
        f.close()
        filename = filename.split(' ')[1].split('\"')[1]
    except:
        print "Failed to load checkpoint information file"
        exit()

    try:
        gs_init = int(filename.split('-')[1])
        ckpt = np.load(os.path.join('results', model, dataset,  experiment_name, 'checkpoints',filename+'.npy'))
    except:
        print "Failed to load saved checkpoint numpy file"
        exit()

    try:
        data_file = open(os.path.join('results', model, dataset, experiment_name, 'checkpoints', filename+'.dat'), 'r')
        data_str = data_file.readlines()
        for data in data_str:
            data_name, data_value = data.split('-')
            if data_name == "lr":
                lr_init = float(data_value)
        data_file.close()

        return ckpt, gs_init, lr_init
    except:
        print "Failed to extract checkpoint data"
        exit()

def save_checkpoint(sess, model, dataset, experiment_name, lr, gs):
    filename = 'checkpoint-'+str(gs)

    c_file = open(os.path.join('results', model, dataset, experiment_name, 'checkpoints','checkpoint'), 'w')
    c_file.write('model_checkpoint_path: "'+filename+'"')
    c_file.close()

    data_file = open(os.path.join('results', model, dataset, experiment_name, 'checkpoints', filename+'.dat'), 'w')
    data_file.write('lr-'+str(lr)+'\n')
    data_file.close()
    data_dict = {}
    for var in tf.global_variables():
        layers = var.name.split('/')
        data_dict = _add_tensor(data_dict, layers, sess.run(var))
    np.save(os.path.join('results', model, dataset,  experiment_name, 'checkpoints',filename+'.npy'), data_dict)


def _add_tensor(data_dict, keys_list, data):
	if len(keys_list) == 0:
		return data

	else:
		try:
			curr_data_dict = data_dict[keys_list[0]]
		except:
			curr_data_dict = {}
		data_dict[keys_list[0]] = _add_tensor(curr_data_dict, keys_list[1:], data)
		return data_dict




def _assign_tensors(sess, curr_dict, tensor_name):
    try:
        if type(curr_dict) == type({}):
            for key in curr_dict.keys():
                _assign_tensors(sess, curr_dict[key], tensor_name+'/'+key)
        else:
            if ':' not in tensor_name:
                tensor_name = tensor_name + ':0'
            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(tensor_name), curr_dict))

    except:
        if 'Momentum' not in tensor_name:
            print "Notice: Tensor " + tensor_name + " could not be assigned properly. The tensors' default initializer will be used if possible. Verify the shape and name of the tensor."




def initialize_from_dict(sess, data_dict):
    print 'Initializng model weights...'
    try:
        data_dict = data_dict.tolist()
        for key in data_dict.keys():
            _assign_tensors(sess, data_dict[key], key)

    except:
        print "Error: Failed to initialize saved weights. Ensure naming convention in saved weights matches the defined model."
        exit()

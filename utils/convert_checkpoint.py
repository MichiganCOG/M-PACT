import os
import argparse
import numpy      as np
import tensorflow as tf



def checkpoint_filename(model, dataset, exp_name):
    """
    Return checkpoint file name for chosen model, dataset and experiment name combination
    Args:
        :model: String indicating name of model
        :dataset: String indicating dataset
        :exp_name: String indicating name of experiment for selected combination of model and dataset

    Returns:
        String reflecting checkpoint filename
    """
    checkpoints_file = os.path.join('results', model, dataset, exp_name, 'checkpoints', 'checkpoint')
    f = open(checkpoints_file, 'r')
    filename = f.readline()
    f.close()
    filename = filename.split(' ')[1].split('\"')[1]
    return filename


def add_tensor(tensor, keys_list, original_key, reader):
    """
    Return contents of selected file
    Args:
        :tensor:       Dictionary containing name and model parameters
        :keys_list:    String list indiciating depth of keys
        :original_key: Original key from meta graph
        :reader:       CheckpointReader instance

    Returns:
        Dictionary of Model parameters
    """
    if len(keys_list) == 0:
        return reader.get_tensor(original_key)

    else:
        try:
            curr_tensor = tensor[keys_list[0]]

        # END TRY

    # END IF

	except:
            curr_tensor = {}

    # END EXCEPT

	tensor[keys_list[0]] = add_tensor(curr_tensor, keys_list[1:], original_key, reader)
    return tensor


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--model', action = 'store')
	parser.add_argument('--dataset', action = 'store')
	parser.add_argument('--expName', action = 'store')

	args = parser.parse_args()

	model = args.model
	dataset = args.dataset
	expName = args.expName

	checkpoint_name = checkpoint_filename(model, dataset, expName)

	checkpoint_path = os.path.join('results', model, dataset, expName, 'checkpoints', checkpoint_name)

	reader = tf.train.NewCheckpointReader(checkpoint_path)
	v_map = reader.get_variable_to_shape_map()
	tensors = {}

	for key in v_map.keys():
		if "Momentum" not in key:
			key_list = key.split('/')
			tensors = add_tensor(tensors, key_list, key, reader)

        # END IF

    # END FOR

    np.save(os.path.join('results', model, dataset, expName, 'checkpoints', checkpoint_name + '.npy'), tensors)
	f = open(os.path.join('results', model, dataset, expName, 'checkpoints', checkpoint_name + '.dat' ), 'w')
	f.write('lr:0.001')
	f.close()

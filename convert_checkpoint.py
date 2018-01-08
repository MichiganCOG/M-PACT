import tensorflow as tf
import numpy as np
import argparse
import os



def checkpoint_filename(model, dataset, expName):
	checkpoints_file = os.path.join('results', model, dataset, expName, 'checkpoints', 'checkpoint')
	f = open(checkpoints_file, 'r')
	filename = f.readline()
	f.close()
	filename = filename.split(' ')[1].split('\"')[1]
	return filename






def add_tensor(tensor, keys_list, original_key, reader):
	if len(keys_list) == 0:
		return reader.get_tensor(original_key)

	else:
		try:
			curr_tensor = tensor[keys_list[0]]
		except:
			curr_tensor = {}
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

	np.save(os.path.join('results', model, dataset, expName, 'checkpoints', checkpoint_name + '.npy'), tensors)

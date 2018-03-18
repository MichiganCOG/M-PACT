import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input', action='store', type=str, help='name of input pbs outfile')
parser.add_argument('--output', action='store', type=str, help='name of output numpy file')

args = parser.parse_args()
input_name = args.input
output_name = args.output + '.npy'

f = open(input_name, 'r')
lines = f.readlines()
f.close()
names = []
labels = []
preds = []

for line in lines:
	if line[:5] == 'vidNa':
		names.append(line.split(' ')[2][:-1])
	elif line[:5] == 'predi':
		preds.append(line.split(' ')[-1][:-1])
	elif line[:5] == 'label':
		labels.append(line.split(' ')[-1][:-1])
names = names[37830:]
labels = labels[37830:]
preds = preds[37830:]
np.save(output_name, np.array([preds, labels, names]).T)
print np.array([preds, labels, names]).T.shape

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

output = np.array(eval(lines[-10][:-1]))

np.save(output_name, output)
print output.shape

#names = []
#labels = []
#preds = []
#import pdb; pdb.set_trace()
#for line in lines:
#	if line[:5] == 'vidNa':
#		names.append(line.split(' ')[2][:-1])
#	elif line[:5] == 'predi':
#		preds.append(int(line.split(' ')[-1][:-1]))
#	elif line[:5] == 'label':
#		labels.append(int(line.split(' ')[-1][:-1]))
#import pdb; pdb.set_trace()
#names = names[15300:]
#labels = labels[15300:]
#preds = preds[15300:]
#np.save(output_name, np.array([preds, labels, names]).T)
#print np.array([preds, labels, names]).T.shape

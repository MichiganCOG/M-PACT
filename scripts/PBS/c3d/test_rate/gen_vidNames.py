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

for line in lines:
	if line[:5] == 'vidNa':
		names.append(line.split(' ')[2][:-1])

names = names[15300:]
np.save(output_name, np.array(names))

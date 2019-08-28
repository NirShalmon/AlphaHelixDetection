'''
A script to be run on chimera, to create raw protein electron density maps and helix data
'''

from chimera import runCommand as rc
import os
import sys


if '--help' in sys.argv:
    raise Exception('run open get_raw_data.py proteins_directory output_directory')


if len(sys.argv) != 4:
    raise Exception('Bad parameters. use --help')

resolution = int(sys.argv[1])

protein_dir = os.path.join(os.getcwd(),sys.argv[2])
output_dir = os.path.join(os.getcwd(),sys.argv[3])

# create the output directory if neccesary
try:
    os.makedirs(output_dir)
except:
    pass


# process all pdb files using chimera
for root, dirs, files in os.walk(protein_dir):
    for filename in files:
        if not filename.endswith('.pdb'):
            continue
        protein_name = filename[:-4]
        protein_path = os.path.join(root, filename)

        rc('open ' + protein_path)  # open protein file
        rc('molmap #0 ' + str(resolution) + ' grid 1 model #1')  # generate electron density map
        rc('volume #1 save ' + os.path.join(output_dir, protein_name + '.mrc'))  # save that map
        rc('molmap helix ' + str(resolution) + ' grid 1 model #2 onGrid #1')#generate heix data, with the same grid size
        rc('volume #2 save ' + os.path.join(output_dir, protein_name + '_helix.mrc'))  # save the helix data
        rc('close all')  #

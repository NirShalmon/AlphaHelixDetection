"""
Script that runs inside Chimera and synthesizes Cryo-EM map and helix mask for proteins (.pdb files)
Editor: Nir Shalmon
"""

from chimera import runCommand as rc
import os
import sys


if '--help' in sys.argv:
    raise Exception('run open get_raw_data.py resolution proteins_directory output_directory')


if len(sys.argv) != 4:
    raise Exception('Bad parameters. use --help')

resolution = int(sys.argv[1])


protein_dir = os.path.join(os.getcwd(), sys.argv[2])
output_dir = os.path.join(os.getcwd(), sys.argv[3])

# Create the output directory if necessary
try:
    os.makedirs(output_dir)
except:
    pass


# Process all pdb files using chimera
for root, dirs, files in os.walk(protein_dir):
    for filename in files:
        if not filename.endswith('.pdb'):
            continue
        protein_name = filename[:-4]
        protein_path = os.path.join(root, filename)

        rc('open ' + protein_path)  # Open protein file
        rc('molmap #0 ' + str(resolution) + ' grid 1 model #1')  # Generate electron density map
        rc('volume #1 save ' + os.path.join(output_dir, protein_name + '.mrc'))  # Save that map
        rc('molmap helix ' + str(resolution) + ' grid 1 model #2 onGrid #1')  # Generate heix data, with the same grid size
        rc('volume #2 save ' + os.path.join(output_dir, protein_name + '_helix.mrc'))  # Save the helix data
        rc('close all')  # Close the open proteins

'''
A script to be run on chimera, to create raw protein electron density maps and helix data
'''

from chimera import runCommand as rc
import os
import sys


resolution = 4 #in angstram. Could be moved to a console parameter

workshop_dir = '/home/nir/workshop' # Needs to be changed to run on other systems
os.chdir(workshop_dir)

protein_dir = os.path.join(workshop_dir,'proteins_pdb') #Could be moved to a console parameter
output_dir = os.path.join(workshop_dir,'raw_data') #Could be moved to a console parameter

#create the output directory if neccesary
try:
	os.makedirs(output_dir)
except:
	pass

#process all pdb files using chimera
for filename in os.listdir(protein_dir):
	if not filename.endswith('.pdb'):
		continue

	protein_name = filename[:-4]
	protein_path = os.path.join(protein_dir,filename) 

	rc('open ' + protein_path) #open protein file
	rc('molmap #0 4 grid 2 model #1') #generate electron density map
	rc('volume #1 save ' + os.path.join(output_dir,protein_name+'.mrc')) #save that map
	rc('molmap helix 4 grid 2 model #2 onGrid #1') #generate heix data, with the same grid size
	rc('volume #2 save ' + os.path.join(output_dir,protein_name+'_helix.mrc')) #save the helix data
	rc('close all') #
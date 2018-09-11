"""Code for extract supersets given a classification matlab file.
Specifically, here we give as input the classification output of the function 
wma_segSuperset_pArcTPC.m or wma_segSuperset_MdLF.m, which classify streamlines
into two classes of supersets, with indeces 1 (left) and 2 (right).
"""

import scipy.io as sio
import numpy as np
import nibabel as nib
import argparse
import sys


def extract_supersets_from_classification(tractogram_filename, tag):

	# Load the inputs
	tractogram = nib.streamlines.load(tractogram_filename)
	tractogram = tractogram.streamlines
	subjID = tractogram_filename[-16:-10]
	classification = '%s_%s_index.mat' %(subjID, tag)
	matlabfile = sio.loadmat(classification)
	indeces = np.array(matlabfile['index'])	

	print("Extracting supersets...")
	idx_Left_superset = []
	idx_Right_superset = []

	for i in range(len(tractogram)):
		if indeces[i] == 1:
			idx_Left_superset.append(i)
		elif indeces[i] == 2:
			idx_Right_superset.append(i)			

	Left_superset = tractogram[idx_Left_superset]
	print("Left_%s_superset has %s streamlines." %(tag,len(Left_superset)))
	Right_superset = tractogram[idx_Right_superset]
	print("Right_%s_superset has %s streamlines." %(tag,len(Right_superset)))

	# Saving supersets indeces
	np.save('sub-%s_Left_%s_superset_idx.npy' %(subjID,tag), idx_Left_superset)
	np.save('sub-%s_Right_%s_superset_idx.npy' %(subjID,tag), idx_Right_superset)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-tractogram', nargs='?', const=1, default='',
	                    help='The tractogram filename')
	parser.add_argument('-tag', nargs='?',  const=1, default='',
	                    help='The tag of the classification matlab structure')  
	parser.add_argument('-tag1', nargs='?',  const=1, default='',
	                    help='The tag of another classification matlab structure')          
	args = parser.parse_args()

	extract_supersets_from_classification(args.tractogram, args.tag)	

	if args.tag1:
		extract_supersets_from_classification(args.tractogram, args.tag1)
	                            
	sys.exit()    

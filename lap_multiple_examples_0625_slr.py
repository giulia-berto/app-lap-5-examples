""" Bundle segmentation with Rectangular Linear Assignment Problem.

	See Sharmin et al., 'White Matter Tract Segmentation as Multiple 
	Linear Assignment Problems', Fronts. Neurosci., 2017.
"""

import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
import time
from nibabel.streamlines import load, save
from tractograms_slr_0625 import tractograms_slr
from lap_single_example_0625_slr import lap_single_example, save_bundle
from dipy.tracking.utils import length


def ranking_schema(superset_estimated_target_tract_idx, superset_estimated_target_tract_cost):
    """ Rank all the extracted streamlines estimated by the LAP with multiple examples   
    according to the number of times that they were selected and the total cost. 
    """
    idxs = np.unique(superset_estimated_target_tract_idx)
    how_many_times_selected = np.array([(superset_estimated_target_tract_idx == idx).sum() for idx in idxs])
    how_much_cost = np.array([((superset_estimated_target_tract_idx == idx)*superset_estimated_target_tract_cost).sum() for idx in idxs])
    ranking = np.argsort(how_many_times_selected)[::-1]
    tmp = np.unique(how_many_times_selected)[::-1]
    for i in tmp:
        tmp1 = (how_many_times_selected == i)
        tmp2 = np.where(tmp1)[0]
        if tmp2.size > 1:
            tmp3 = np.argsort(how_much_cost[tmp2])
            ranking[how_many_times_selected[ranking]==i] = tmp2[tmp3]
 
    return idxs[ranking]


def lap_multiple_examples(moving_tractograms_dir, static_tractogram, ex_dir, out_filename, stat_id):
	"""Code for LAP from multiple examples.
	"""
	moving_tractograms = os.listdir(moving_tractograms_dir)
	moving_tractograms.sort()
	examples = os.listdir(ex_dir)
	examples.sort()

	nt = len(moving_tractograms)
	ne = len(examples)

	if nt != ne:
		print("Error: number of moving tractograms differs from number of example bundles.")
		sys.exit()
	else:	
		result_lap = []
		for i in range(nt):
			moving_tractogram = '%s/%s' %(moving_tractograms_dir, moving_tractograms[i])
			example = '%s/%s' %(ex_dir, examples[i])
			mov_id = moving_tractograms[i][0:6]
			tmp = np.array([lap_single_example(moving_tractogram, static_tractogram, example, mov_id, stat_id)])
			result_lap.append(tmp)

		result_lap = np.array(result_lap)
		estimated_bundle_idx = np.hstack(result_lap[:,0,0])
		min_cost_values = np.hstack(result_lap[:,0,1])
		example_bundle_len_med = np.median(np.hstack(result_lap[:,0,2]))

		print("Ranking the estimated streamlines...")
		estimated_bundle_idx_ranked = ranking_schema(estimated_bundle_idx, min_cost_values)                      

		print("Extracting the estimated bundle...")
		estimated_bundle_idx_ranked_med = estimated_bundle_idx_ranked[0:int(example_bundle_len_med)]
		
		save_bundle(estimated_bundle_idx_ranked_med, static_tractogram, out_filename)

		return result_lap


if __name__ == '__main__':

	np.random.seed(0) 

	parser = argparse.ArgumentParser()
	parser.add_argument('-moving_dir', nargs='?', const=1, default='',
	                    help='The moving tractogram directory')
	parser.add_argument('-static', nargs='?',  const=1, default='',
	                    help='The static tractogram filename')
	parser.add_argument('-ex_dir', nargs='?',  const=1, default='',
	                    help='The examples (moving) bundle directory')
	parser.add_argument('-out', nargs='?',  const=1, default='default',
	                    help='The output estimated bundle filename')    
	parser.add_argument('-stat_id', nargs='?',  const=1, default='default',
	                    help='Static subject id')                
	args = parser.parse_args()

	t0=time.time()
	result_lap = lap_multiple_examples(args.moving_dir, args.static, args.ex_dir, args.out, args.stat_id)
	print("Time for computing multiple-lap = %s seconds" %(time.time()-t0))

	np.save('result_lap', result_lap)

	sys.exit()    

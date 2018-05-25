""" Code to compute the dissimilarity representation of a tractogram.

"""

from __future__ import division
import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
from nibabel.streamlines import load
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
import time
import pickle


def compute_kdtree_and_dr_tractogram(tractogram, num_prototypes=None, distance=bundles_distances_mam):
    """Compute the dissimilarity representation of the target tractogram and 
    build the kd-tree.
    """
    t0 = time.time()
    tractogram = np.array(tractogram, dtype=np.object)
    print("Computing dissimilarity matrices...")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012."
              % num_prototypes)
    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance=bundles_distances_mam,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    prototypes = tractogram[prototype_idx]
    print("Building the KD-tree of tractogram.")
    kdt = KDTree(dm_tractogram)
    print("Time spent to compute the dissimilarity representation of the tractogram: %i minutes" %((time.time()-t0)/60))
    return kdt, prototypes



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-static', nargs='?', const=1, default='',
	                    help='The moving tractogram filename')                  
	args = parser.parse_args()

	static_tractogram = nib.streamlines.load(args.static)
	static_tractogram = static_tractogram.streamlines

	kdt, prototypes = compute_kdtree_and_dr_tractogram(static_tractogram)	

	#Saving files
	kdt_filename='kdt'
	pickle.dump(kdt, open(kdt_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
	np.save('prototypes', prototypes)
	                            
	sys.exit()    


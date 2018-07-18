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
import pickle
import json
from os.path import isfile
from nibabel.streamlines import load
from tractograms_slr_0625 import tractograms_slr
from dipy.tracking.streamline import apply_affine
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import StreamlineLinearRegistration
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.utils import length
from sklearn.neighbors import KDTree
from dipy.viz import fvtk


try:
    from linear_assignment import LinearAssignment
except ImportError:
    print("WARNING: Cythonized LAPJV not available. Falling back to Python.")
    print("WARNING: See README.txt")
    from linear_assignment_numpy import LinearAssignment


def resample_tractogram(tractogram, step_size):
    """Resample the tractogram with the given step size.
    """
    lengths=list(length(tractogram))
    tractogram_res = []
    for i, f in enumerate(tractogram):
	nb_res_points = np.int(np.floor(lengths[i]/step_size))
	tmp = set_number_of_points(f, nb_res_points)
	tractogram_res.append(tmp)
    return tractogram_res


def local_slr(moving_tractogram, static_tractogram):
    """Perform local SLR.
    """
    print("Resampling tractograms with step size = 0.625 mm") 
    static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)	
    static_tractogram = static_tractogram_res
    moving_tractogram_res = resample_tractogram(moving_tractogram, step_size=0.625)	
    moving_tractogram = moving_tractogram_res

    print("Set parameters as in Garyfallidis et al. 2015.") 
    threshold_length = 40.0 # 50mm / 1.25
    qb_threshold = 16.0  # 20mm / 1.25 
    nb_res_points = 20

    print("Performing QuickBundles of static tractogram and resampling...")
    st = np.array([s for s in static_tractogram if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    st_clusters = [cluster.centroid for cluster in qb.cluster(st)]
    st_clusters = set_number_of_points(st_clusters, nb_res_points)

    print("Performing QuickBundles of moving tractogram and resampling...")
    mt = np.array([s for s in moving_tractogram if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    mt_clusters = [cluster.centroid for cluster in qb.cluster(mt)]
    mt_clusters = set_number_of_points(mt_clusters, nb_res_points)

    print("Performing Linear Registration...")
    srr = StreamlineLinearRegistration()
    srm = srr.optimize(static=st_clusters, moving=mt_clusters)

    print("Affine transformation matrix with Streamline Linear Registration:")
    affine = srm.matrix
    print('%s' %affine)

    return affine


def compute_kdtree_and_dr_tractogram(tractogram, num_prototypes=None):
    """Compute the dissimilarity representation of the target tractogram and 
    build the kd-tree.
    """
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
    return kdt, prototypes    


def RLAP(kdt, k, dm_source_tract, source_tract, tractogram, distance):
    """Code for Rectangular Linear Assignment Problem.
    """
    tractogram = np.array(tractogram, dtype=np.object)
    D, I = kdt.query(dm_source_tract, k=k)
    superset = np.unique(I.flat)
    np.save('superset_idx', superset)
    with open('config.json') as f:
        data = json.load(f)
	if data["local_slr"] == True:
	    print("Computing local SLR")
	    local_affine = local_slr(source_tract, tractogram[superset])
	    source_tract_aligned = np.array([apply_affine(local_affine, s) for s in source_tract])
	    source_tract = source_tract_aligned
    print("Computing the cost matrix (%s x %s) for RLAP... " % (len(source_tract),
                                                             len(superset)))
    cost_matrix = dissimilarity(source_tract, tractogram[superset], distance)
    print("Computing RLAP with LAPJV...")
    assignment = LinearAssignment(cost_matrix).solution
    estimated_bundle_idx = superset[assignment]
    min_cost_values = cost_matrix[np.arange(len(cost_matrix)), assignment]

    return estimated_bundle_idx, min_cost_values


def show_tracts(estimated_target_tract, target_tract):
	"""Visualization of the tracts.
	"""
	ren = fvtk.ren()
	fvtk.add(ren, fvtk.line(estimated_target_tract, fvtk.colors.green,
	                        linewidth=1, opacity=0.3))
	fvtk.add(ren, fvtk.line(target_tract, fvtk.colors.white,
	                        linewidth=2, opacity=0.3))
	fvtk.show(ren)
	fvtk.clear(ren)


def lap_single_example(moving_tractogram, static_tractogram, example):
	"""Code for LAP from a single example.
	"""
	with open('config.json') as f:
            data = json.load(f)
	    k = data["k"]
	    ANTs = data["ANTs"]
	distance_func = bundles_distances_mam

	example_bundle = nib.streamlines.load(example)
	example_bundle = example_bundle.streamlines
	example_bundle_res = resample_tractogram(example_bundle, step_size=0.625)
	
	if ANTs == True:
		print("Data already aligned with ANTs")
		example_bundle_aligned = example_bundle_res
	else:
		print("Computing the affine slr transformation.")
		affine = tractograms_slr(moving_tractogram, static_tractogram)
		print("Applying the affine to the example bundle.")
		example_bundle_aligned = np.array([apply_affine(affine, s) for s in example_bundle_res])
	
	print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
	static_tractogram = nib.streamlines.load(static_tractogram)
	static_tractogram = static_tractogram.streamlines
	static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)	
	static_tractogram = static_tractogram_res
	if isfile('prototypes.npy') & isfile('kdt'):
		print("Retrieving past results for kdt and prototypes.")
		kdt_filename='kdt'
		kdt = pickle.load(open(kdt_filename))
		prototypes = np.load('prototypes.npy')
	else:
		kdt, prototypes = compute_kdtree_and_dr_tractogram(static_tractogram)
		#Saving files
		kdt_filename='kdt'
		pickle.dump(kdt, open(kdt_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
		np.save('prototypes', prototypes)

	print("Compute the dissimilarity of the aligned example bundle with the prototypes of target tractogram.")
	example_bundle_aligned = np.array(example_bundle_aligned, dtype=np.object)
	dm_example_bundle_aligned = distance_func(example_bundle_aligned, prototypes)

	print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
	estimated_bundle_idx, min_cost_values = RLAP(kdt, k, dm_example_bundle_aligned, example_bundle_aligned, static_tractogram, distance_func)

	return estimated_bundle_idx, min_cost_values, len(example_bundle)


def save_bundle(estimated_bundle_idx, static_tractogram, out_filename):

	extension = os.path.splitext(out_filename)[1]
	static_tractogram = nib.streamlines.load(static_tractogram)
	aff_vox_to_ras = static_tractogram.affine
	voxel_sizes = static_tractogram.header['voxel_sizes']
	dimensions = static_tractogram.header['dimensions']
	static_tractogram = static_tractogram.streamlines
	static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)
	static_tractogram = np.array(static_tractogram_res, dtype=np.object)
	estimated_bundle = static_tractogram[estimated_bundle_idx]
	
	if extension == '.trk':
		print("Saving bundle in %s" % out_filename)
		
		# Creating header
		hdr = nib.streamlines.trk.TrkFile.create_empty_header()
		hdr['voxel_sizes'] = voxel_sizes
		hdr['voxel_order'] = 'LAS'
		hdr['dimensions'] = dimensions
		hdr['voxel_to_rasmm'] = aff_vox_to_ras 

		# Saving bundle
		t = nib.streamlines.tractogram.Tractogram(estimated_bundle, affine_to_rasmm=aff_vox_to_ras)
		nib.streamlines.save(t, out_filename, header=hdr)
		print("Bundle saved in %s" % out_filename)

	elif extension == '.tck':
		print("Saving bundle in %s" % out_filename)

		# Creating header
		hdr = nib.streamlines.tck.TckFile.create_empty_header()
		hdr['voxel_sizes'] = voxel_sizes
		hdr['dimensions'] = dimensions
		hdr['voxel_to_rasmm'] = aff_vox_to_ras

		# Saving bundle
		t = nib.streamlines.tractogram.Tractogram(estimated_bundle, affine_to_rasmm=np.eye(4))
		nib.streamlines.save(t, out_filename, header=hdr)
		print("Bundle saved in %s" % out_filename)

	else:
		print("%s format not supported." % extension)	


if __name__ == '__main__':

	np.random.seed(0) 

	parser = argparse.ArgumentParser()
	parser.add_argument('-moving', nargs='?', const=1, default='',
	                    help='The moving tractogram filename')
	parser.add_argument('-static', nargs='?',  const=1, default='',
	                    help='The static tractogram filename')
	parser.add_argument('-ex', nargs='?',  const=1, default='',
	                    help='The example (moving) bundle filename')  
	parser.add_argument('-out', nargs='?',  const=1, default='',
	                    help='The output estimated bundle filename')                               
	args = parser.parse_args()

	result_lap = lap_single_example(args.moving, args.static, args.ex)

	np.save('result_lap', result_lap)

	if args.out:
		estimated_bundle_idx = result_lap[0]
		save_bundle(estimated_bundle_idx, args.static, args.out)

	sys.exit()    


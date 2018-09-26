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
import time
import ntpath
from os.path import isfile
from nibabel.streamlines import load
from tractograms_slr_0625 import tractograms_slr
from dipy.tracking.streamline import apply_affine
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import StreamlineLinearRegistration
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from compute_streamline_measures import streamlines_idx, compute_superset
from endpoints_distance import bundles_distances_endpoints_fastest
from dipy.tracking.utils import length
from dipy.tracking import metrics as tm
from sklearn.neighbors import KDTree
from dipy.viz import fvtk
from nibabel.affines import apply_affine 
from scipy.spatial.distance import cdist


try:
    from linear_assignment import LinearAssignment
except ImportError:
    print("WARNING: Cythonized LAPJV not available. Falling back to Python.")
    print("WARNING: See README.txt")
    from linear_assignment_numpy import LinearAssignment


def bundle2roi_distance(bundle, roi_mask, distance='euclidean'):
	"""Compute the minimum euclidean distance between a
	   set of streamlines and a ROI nifti mask.
	"""
	data = roi_mask.get_data()
	affine = roi_mask.affine
	roi_coords = np.array(np.where(data)).T
	x_roi_coords = apply_affine(affine, roi_coords)
	result=[]
	for sl in bundle:                                                                                  
		d = cdist(sl, x_roi_coords, distance)
		result.append(np.min(d)) 
	return result


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


def compute_lap_matrices(superset_idx, source_tract, tractogram, roi1, roi2):
	"""Code for computing the inputs to the MODIFIED Rectangular Linear Assignment Problem.
	"""
	distance = bundles_distances_mam
	tractogram = np.array(tractogram, dtype=np.object)

	print("Computing the distance matrix (%s x %s) for RLAP with %s... " % (len(source_tract), len(superset_idx), distance))
	t0=time.time()
	distance_matrix = dissimilarity(source_tract, tractogram[superset_idx], distance)
	print("Time for computing the distance matrix = %s seconds" %(time.time()-t0))
	
	print("Computing the terminal points matrix (%s x %s) for RLAP... " % (len(source_tract), len(superset_idx)))
    	t1=time.time()
    	terminal_matrix = bundles_distances_endpoints_fastest(source_tract, tractogram[superset_idx])
    	print("Time for computing the terminal points matrix = %s seconds" %(time.time()-t1))

	print("Computing the anatomical matrix (%s x %s) for RLAP... " % (len(source_tract), len(superset_idx)))
	t2=time.time()
	roi1_dist = bundle2roi_distance(tractogram[superset_idx], roi1)
	roi2_dist = bundle2roi_distance(tractogram[superset_idx], roi2)
	anatomical_vector = np.add(roi1_dist, roi2_dist)
	anatomical_matrix = np.zeros((len(source_tract), len(superset_idx)))
	for i in range(len(source_tract)):
		anatomical_matrix[i] = anatomical_vector
	print("Time for computing the anatomical matrix = %s seconds" %(time.time()-t2))

	#normalize matrices
	distance_matrix = (distance_matrix-np.min(distance_matrix))/(np.max(distance_matrix)-np.min(distance_matrix))
	terminal_matrix = (terminal_matrix-np.min(terminal_matrix))/(np.max(terminal_matrix)-np.min(terminal_matrix))
	anatomical_matrix = (anatomical_matrix-np.min(anatomical_matrix))/(np.max(anatomical_matrix)-np.min(anatomical_matrix))

	return distance_matrix, terminal_matrix, anatomical_matrix


def RLAP_modified(distance_matrix, terminal_matrix, anatomical_matrix, superset_idx, g, alpha):
    """Code for MODIFIED Rectangular Linear Assignment Problem.
    """
    print("Computing cost matrix.")
    cost_matrix = distance_matrix + g * terminal_matrix + alpha * anatomical_matrix
    print("Computing RLAP with LAPJV...")
    t0=time.time()
    assignment = LinearAssignment(cost_matrix).solution
    estimated_bundle_idx = superset_idx[assignment]
    min_cost_values = cost_matrix[np.arange(len(cost_matrix)), assignment]
    print("Time for computing the solution to the assignment problem = %s seconds" %(time.time()-t0))

    return estimated_bundle_idx, min_cost_values


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


def lap_single_example(moving_tractogram, static_tractogram, example):
	"""Code for LAP from a single example.
	"""
	with open('config.json') as f:
            data = json.load(f)
	    k = data["k"]
	distance_func = bundles_distances_mam

	subjID = ntpath.basename(static_tractogram)[0:6]
	tract_name = ntpath.basename(example)[7:-10]
	exID = ntpath.basename(example)[0:6]

	example_bundle = nib.streamlines.load(example)
	example_bundle = example_bundle.streamlines
	example_bundle_res = resample_tractogram(example_bundle, step_size=0.625)
	
	print("Data already aligned with ANTs")
	example_bundle_aligned = example_bundle_res
	
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

	print("Computing superset with k = %s" %k)
	superset_idx = compute_superset(example_bundle_aligned, kdt, prototypes, k=k)

	print("Loading the two-waypoint ROIs of the target...")
	table_filename = 'ROIs_labels_dictionary.pickle'
	table = pickle.load(open(table_filename))
	roi1_lab = table[tract_name].items()[0][1]
	roi1_filename = 'aligned_ROIs/sub-%s_var-AFQ_lab-%s_roi.nii.gz' %(subjID, roi1_lab)
	roi1 = nib.load(roi1_filename)
	roi2_lab = table[tract_name].items()[1][1]
	roi2_filename = 'aligned_ROIs/sub-%s_var-AFQ_lab-%s_roi.nii.gz' %(subjID, roi2_lab)
	roi2 = nib.load(roi2_filename)
	
	distance_matrix, terminal_matrix, anatomical_matrix = compute_lap_matrices(superset_idx, example_bundle_aligned, static_tractogram, roi1, roi2)

	g = 1 
	alpha = 1 

	print("Using g = %s and alpha = %s" %(g,alpha))
	estimated_bundle_idx, min_cost_values = RLAP_modified(distance_matrix, terminal_matrix, anatomical_matrix, superset_idx, g, alpha)

	return estimated_bundle_idx, min_cost_values, len(example_bundle)
	


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

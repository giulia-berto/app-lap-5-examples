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

def curvature_diff(streamlines_A, streamlines_B):
    curvature_A = np.zeros(len(streamlines_A))
    curvature_B = np.zeros(len(streamlines_B))
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        curvature_A[i] = tm.mean_curvature(sa)
    for j, sb in enumerate(streamlines_B):
        curvature_B[j] = tm.mean_curvature(sb)
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result[i, j] = abs(curvature_A[i]-curvature_B[j])
	#print("Row %s done." %i)
    return result


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


def compute_lap_matrices(superset_idx, source_tract, tractogram, distance, nbp=200):
    """Code for computing the inputs to the MODIFIED Rectangular Linear Assignment Problem.
    """
    tractogram = np.array(tractogram, dtype=np.object)
    if distance=="bundles_distances_mdf":
    	print("Resampling at %s points." %nbp)
    	source_tract_res = np.array([set_number_of_points(s, nb_points=nbp) for s in source_tract])
    	tractogram_res = np.array([set_number_of_points(s, nb_points=nbp) for s in tractogram])
    
    	print("Computing the distance matrix (%s x %s) for RLAP with %s... " % (len(source_tract), len(superset_idx), distance))
    	t0=time.time()
    	distance_matrix = dissimilarity(source_tract_res, tractogram_res[superset_idx], distance)
    	print("Time for computing the distance matrix = %s seconds" %(time.time()-t0))

    else:
	print("Computing the distance matrix (%s x %s) for RLAP with %s... " % (len(source_tract), len(superset_idx), distance))
    	t0=time.time()
    	distance_matrix = dissimilarity(source_tract, tractogram[superset_idx], distance)
    	print("Time for computing the distance matrix = %s seconds" %(time.time()-t0))

        print("Resampling at %s points." %nbp)
    	source_tract_res = np.array([set_number_of_points(s, nb_points=nbp) for s in source_tract])
    	tractogram_res = np.array([set_number_of_points(s, nb_points=nbp) for s in tractogram])

    print("Computing the terminal points matrix (%s x %s) for RLAP... " % (len(source_tract), len(superset_idx)))
    t1=time.time()
    terminal_matrix = bundles_distances_endpoints_fastest(source_tract, tractogram[superset_idx])
    print("Time for computing the terminal points matrix = %s seconds" %(time.time()-t1))
    
    print("Computing the curvature similarity matrix (%s x %s) for RLAP... " % (len(source_tract), len(superset_idx)))
    t1=time.time()
    curvature_matrix = curvature_diff(source_tract_res, tractogram_res[superset_idx])
    print("Time for computing the curvature similarity matrix = %s seconds" %(time.time()-t1))
    
    #normalize matrices
    distance_matrix = (distance_matrix-np.min(distance_matrix))/(np.max(distance_matrix)-np.min(distance_matrix))
    curvature_matrix = (curvature_matrix-np.min(curvature_matrix))/(np.max(curvature_matrix)-np.min(curvature_matrix))
    terminal_matrix = (terminal_matrix-np.min(terminal_matrix))/(np.max(terminal_matrix)-np.min(terminal_matrix))

    return distance_matrix, curvature_matrix, terminal_matrix


def RLAP_modified_terminal_curvature(distance_matrix, curvature_matrix, terminal_matrix, superset_idx, g, h):
    """Code for MODIFIED Rectangular Linear Assignment Problem.
    """
    print("Computing cost matrix.")
    cost_matrix = distance_matrix + g * terminal_matrix + h * curvature_matrix
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

	print("Loading superset idx...")
	subjID = static_tractogram[-16:-10]
	if example[-19:-10] == 'Left_pArc' or example[-18:-10] == 'Left_TPC':
		tag = 'Left_parctpc'
		print("Tag: %s" %tag)
	elif example[-20:-10] == 'Right_pArc' or example[-19:-10] == 'Right_TPC':
		tag = 'Right_parctpc' 
		print("Tag: %s" %tag)
	elif example[-23:-10] == 'Left_MdLF-SPL' or example[-23:-10] == 'Left_MdLF-Ang':
		tag = 'Left_mdlf'
		print("Tag: %s" %tag)
	elif example[-24:-10] == 'Right_MdLF-SPL' or example[-24:-10] == 'Right_MdLF-Ang':
		tag = 'Right_mdlf'
		print("Tag: %s" %tag)
	superset_idx = np.load('supersets_idx/sub-%s_%s_superset_idx.npy' %(subjID, tag))
	
	print("Loading static tractogram...")
	static_tractogram = nib.streamlines.load(static_tractogram)
	static_tractogram = static_tractogram.streamlines
	static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)	
	static_tractogram = static_tractogram_res

	with open('config.json') as f:
            data = json.load(f)
	if data["local_slr"] == True:
	    print("Computing local SLR")
	    local_affine = local_slr(example_bundle_aligned, static_tractogram[superset_idx])
	    source_tract_aligned = np.array([apply_affine(local_affine, s) for s in example_bundle_aligned])
	    example_bundle_aligned = source_tract_aligned

	print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
	distance_matrix, curvature_matrix, terminal_matrix = compute_lap_matrices(superset_idx, example_bundle_aligned, static_tractogram, distance=distance_func)

	with open('config.json') as f:
            data = json.load(f)
	    g = data["g"]
	    h = data["h"]

	print("Using g = %s and h = %s" %(g,h))
	estimated_bundle_idx, min_cost_values = RLAP_modified_terminal_curvature(distance_matrix, curvature_matrix, terminal_matrix, superset_idx, g, h)

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

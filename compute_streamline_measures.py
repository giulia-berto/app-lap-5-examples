"""Compute some streamline measures"""

from __future__ import print_function, division
import numpy as np
import nibabel as nib
from nibabel.streamlines import load
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from lap_single_example import compute_kdtree_and_dr_tractogram
from dissimilarity import compute_dissimilarity, dissimilarity
from sklearn.neighbors import KDTree
from dipy.segment.clustering import QuickBundles
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle
import os
from os.path import isfile


def compute_loss_function(source_tract, ett):
    """Compute the loss function between two tracts. 
    """
    sP = len(source_tract) 
    sQ = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg') 
    L = np.sum(distance_matrix)
    L = L / (sP*sQ)   
    return L     


def bmd(source_tract, ett, nbp=200):
    """Compute the cost function Bundle-based Minimum Distance (BMD) 
    as in [Garyfallidis et al. 2015]. 
    """
    A = len(source_tract) 
    B = len(ett)
    source_tract_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in source_tract])
    ett_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in ett])
    distance_matrix = bundles_distances_mdf(source_tract_res, ett_res)

    min_a = 0
    min_b = 0
    for j in range(A):
        min_a = min_a + np.min(distance_matrix[j])
    for i in range(B):
        min_b = min_b + np.min(distance_matrix[:,i]) 
    BMD = ((min_a/A + min_b/B)**2)/4    

    return BMD     


def compute_bmd(source_tract, ett):
    """Compute the cost function Bundle-based Minimum Distance (BMD) 
    as in [Garyfallidis et al. 2015], but using the mam_avg distance 
    instead of the MDF distance. 
    """
    A = len(source_tract) 
    B = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg')

    min_a = 0
    min_b = 0
    for j in range(A):
        min_a = min_a + np.min(distance_matrix[j])
    for i in range(B):
        min_b = min_b + np.min(distance_matrix[:,i]) 
    BMD = ((min_a/A + min_b/B)**2)/4    

    return BMD 


def compute_loss_and_bmd(source_tract, ett):
    """Compute loss function and BMD.
    """
    A = len(source_tract) 
    B = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg')

    L = np.sum(distance_matrix)
    L = L / (A*B)

    min_a = 0
    min_b = 0
    for j in range(A):
        min_a = min_a + np.min(distance_matrix[j])
    for i in range(B):
        min_b = min_b + np.min(distance_matrix[:,i]) 
    BMD = ((min_a/A + min_b/B)**2)/4    

    return L, BMD 


def compute_centroid(source_tract):
    """Compute the distance of the centroids of two bundles.
    """
    threshold_length = 40.0 # 50mm / 1.25
    qb_threshold = 100.0 #to ensure only one cluster
    nb_res_points = 200

    st = np.array([s for s in source_tract if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    st_clusters = [cluster.centroid for cluster in qb.cluster(st)]
    st_clusters = set_number_of_points(st_clusters, nb_res_points)

    return st_clusters


def compute_centroids_dist(source_tract, ett):
    """Compute the distance of the centroids of two bundles.
    """
    threshold_length = 40.0 # 50mm / 1.25
    qb_threshold = 100.0 #to ensure only one cluster
    nb_res_points = 200

    st = np.array([s for s in source_tract if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    st_clusters = [cluster.centroid for cluster in qb.cluster(st)]
    st_clusters = set_number_of_points(st_clusters, nb_res_points)   

    mt = np.array([s for s in ett if len(s) > threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    mt_clusters = [cluster.centroid for cluster in qb.cluster(mt)]
    mt_clusters = set_number_of_points(mt_clusters, nb_res_points)

    return bundles_distances_mdf(st_clusters, mt_clusters)


def compute_superset(true_tract, kdt, prototypes, k=1000, distance_func=bundles_distances_mam):
    """Compute a superset of the true target tract with k-NN.
    """
    true_tract = np.array(true_tract, dtype=np.object)
    dm_true_tract = distance_func(true_tract, prototypes)
    D, I = kdt.query(dm_true_tract, k=k)
    superset_idx = np.unique(I.flat)
    return superset_idx


def streamlines_idx(target_tract, kdt, prototypes, distance_func=bundles_distances_mam, warning_threshold=1.0e-4):
    """Retrieve indexes of the streamlines of the target tract.
    """
    dm_target_tract = distance_func(target_tract, prototypes)
    D, I = kdt.query(dm_target_tract, k=1)
    if (D > warning_threshold).any():
        print("WARNING (streamlines_idx()): for %s streamlines D > 1.0e-4 !!" % (D > warning_threshold).sum())
    #print(D)
    target_tract_idx = I.squeeze()
    return target_tract_idx 


def compute_y_vectors_lap(estimated_tract_idx, estimated_tract_idx_ranked, true_tract, target_tractogram):
    """Compute y_true and y_score. Here estimated_tract_idx and estimated_tract_idx_ranked refer to the
       estimated tract obtained from LAP single example, or from LAP multiple examples after refinement.
    """ 
    print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
    kdt, prototypes = compute_kdtree_and_dr_tractogram(target_tractogram)

    print("Compute a superset of the true target tract with k-NN.")
    superset_idx = compute_superset(true_tract, kdt, prototypes)

    print("Retrieving indeces of the true_tract")
    true_tract_idx = streamlines_idx(true_tract, kdt, prototypes)

    print("Compute y_true.")
    y_true = np.zeros(len(superset_idx))
    correspondent_idx_true = np.array([np.where(superset_idx==true_tract_idx[i]) for i in range(len(true_tract_idx))])
    y_true[correspondent_idx_true] = 1

    print("Compute y_score.")
    S = len(estimated_tract_idx)
    y_score = S*np.ones(len(superset_idx))
    correspondent_idx_score = np.array([np.where(superset_idx==estimated_tract_idx[i]) for i in range(S)])
    for i in range(S):
        y_score[correspondent_idx_score[i]] = estimated_tract_idx_ranked[i]
    #invert the ranking   
    y_score = abs(y_score-S)

    return y_true, y_score


def compute_roc_curve_lap(candidate_idx_ranked, true_tract, target_tractogram, kdt, prototypes):
    """Compute ROC curve. Here candidate_idx_ranked refers to all the candidate 
       streamlines obtained from LAP multiple examples before refinement.
    """
    print("Retrieving indeces of the true_tract")
    true_tract_idx = streamlines_idx(true_tract, kdt, prototypes)

    print("Compute y_score.")
    y_score = np.arange(len(candidate_idx_ranked),0,-1)

    print("Compute y_true.")
    diff = np.setdiff1d(true_tract_idx, candidate_idx_ranked)

    if (len(diff) != 0):
       print("There are %s/%s streamlines of the true tract that aren't in the superset. Making the superset bigger." %(len(diff), len(true_tract_idx)))
       candidate_idx_ranked = np.concatenate([candidate_idx_ranked, diff])
       y_score = np.concatenate([y_score, np.zeros(len(diff))])

    y_true = np.zeros(len(candidate_idx_ranked))
    correspondent_idx_true = np.array([np.where(candidate_idx_ranked==true_tract_idx[i]) for i in range(len(true_tract_idx))])
    y_true[correspondent_idx_true] = 1

    print("Compute ROC curve and AUC.")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)

    return fpr, tpr, AUC

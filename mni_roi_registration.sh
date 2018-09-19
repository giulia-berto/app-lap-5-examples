#!/bin/bash

SUB_STAT=$1
T1W_STAT=$2
MNI=templates/MNI_JHU_T2.nii.gz

if [ ! -z $3 ]; then
	RUN=$3
	EXT=_$3
else
	EXT=""
fi

#------------------------------------------------------------------------------
# Pathname Settings
#------------------------------------------------------------------------------

CMD=$0
# BIN=$(dirname $CMD)
# SRC=$(cd $BIN/../..; pwd)
# DER=$(basename "${CMD%.*}"${EXT})
PAR="${CMD%.*}${EXT}.sh"
# OUT=$SRC/derivatives/$DER/sub-$SUB

# mkdir -p $SRC/derivatives/$DER
# mkdir -p $SRC/derivatives/$DER/sub-$SUB

OUT=aligned_ROIs
mkdir -p $OUT

#------------------------------------------------------------------------------
# Run Settings
#------------------------------------------------------------------------------

# Required variables
# ROI_ALL: the list of ROIs to be registered from MNI atlas
# ROI_CUT: the threshold value to binarize the ROI map after warping

if [ "$EXT" != "" ]; then
	if [ ! -f $PAR ]; then
	    echo "WARNING: The file with run settings doesn't exist: $PAR"
	else
    	. $PAR
	fi
fi


#------------------------------------------------------------------------------
# Setting filenames
#------------------------------------------------------------------------------

#ANT_BIN=${SRC}/derivatives/ants-registration

#mkdir -p ${ant_BIN}/sub-$SUB
#mkdir -p ${ant_BIN}/sub-$SUB/aligned_rois

ANT_BIN=$ANTSPATH
ANT_PRE=${OUT}/sub-${SUB_STAT}_var-ant_
ANT_WARP=${ANT_PRE}1Warp.nii.gz 
ANT_AFF=${ANT_PRE}0GenericAffine.mat


#------------------------------------------------------------------------------
# ROI Registration from MNI JUH
#------------------------------------------------------------------------------

# Warp Computation
echo "Computing the warp..."
if [ ! -f ${ANT_PRE}1Warp.nii.gz ]; then
    ${ANT_BIN}/antsRegistrationSyNQuick.sh -d 3 \
	-f $T1W_STAT -m $MNI -t s -o $ANT_PRE
fi

# ROI Warping
echo "Applying the warp..."
for ROI in $ROI_ALL; do
    LAB=$(eval echo "\$ROI_$ROI")
    NII=templates/MNI_JHU_tracts_ROIs/${LAB}.nii.gz
    if [ ! -f $NII ]; then
	echo "WARNING the atlas ROI called $LAB is not available."
	exit
    else
	WarpImageMultiTransform 3  \
	    $NII \
	    $OUT/sub-${SUB_STAT}_var-${RUN}_lab-${LAB//_/}_roi.nii.gz \
	    $ANT_WARP $ANT_AFF
    fi
done

# ROI Thresholding
echo "Thresholding ROIs..."
for ROI in $ROI_ALL; do
	if [[ $ROI == CST* ]]; then 
		ROI_CUT=0.15; 
	fi
    LAB=$(eval echo "\$ROI_$ROI")
    ROI_HCP=${OUT}/sub-${SUB_STAT}_var-${RUN}_lab-${LAB//_/}_roi.nii.gz
    fslmaths ${ROI_HCP} -thr ${ROI_CUT} -bin ${ROI_HCP}
done


#------------------------------------------------------------------------------
# IU Karst settings
#------------------------------------------------------------------------------

#PBS -k o 
#PBS -l nodes=1:ppn=16,mem=16000mb,walltime=10:00:00 
#PBS -M pavesani@iu.edu
#PBS -m abe
#PBS -N template_script
#PBS -j oe

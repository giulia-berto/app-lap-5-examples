#!/bin/bash

SUB_MOVE=$1
T1W_MOVE=$2
SUB_STAT=$3
T1W_STAT=$4


#------------------------------------------------------------------------------
# Renaming files
#------------------------------------------------------------------------------

echo "renaming files"
cp $T1W_MOVE ./sub-${SUB_MOVE}_space-dwi_T1w.nii.gz;
cp $T1W_STAT ./sub-${SUB_STAT}_space-dwi_T1w.nii.gz

T1W_ANT_PRE=sub-${SUB_MOVE}_space_${SUB_STAT}_var-t1w_


#------------------------------------------------------------------------------
# ANTS Registration of structural images
#------------------------------------------------------------------------------

# Warp Computation
#ANTS 3 -m CC[${T1W_STAT},${T1W_MOVE},1,5] -t SyN[0.5] \
#    -r Gauss[2,0] -o $T1W_ANT_PRE -i 30x90x20 --use-Histogram-Matching

antsRegistrationSyNQuick.sh -d 3 \
     -f $T1W_STAT -m $T1W_MOVE -t s -o $T1W_ANT_PRE


#------------------------------------------------------------------------------
# ANTS Registration of structural images
#------------------------------------------------------------------------------

mv ${T1W_ANT_PRE}Warp.nii.gz ${T1W_ANT_PRE}warp.nii.gz
mv ${T1W_ANT_PRE}InverseWarp.nii.gz ${T1W_ANT_PRE}invwarp.nii.gz
mv ${T1W_ANT_PRE}Affine.txt ${T1W_ANT_PRE}affine.txt

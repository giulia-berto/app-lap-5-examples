#!/bin/bash

SUB_MOVE=$1
T1W_MOVE=$2
SUB_STAT=$3
T1W_STAT=$4


#------------------------------------------------------------------------------
# REGISTRATION OF TRACTS
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Setting filenames
#------------------------------------------------------------------------------

WARP_PRE=sub-${SUB_MOVE}_space_${SUB_STAT}_var-ant_warp
WARP_INI=sub-${SUB_MOVE}_space_${SUB_STAT}_var-ant_warp-i[].nii.gz
WARP_TMP=sub-${SUB_MOVE}_space_${SUB_STAT}_var-ant_warp-t[].nii.gz
WARP_T1W=sub-${SUB_MOVE}_space_${SUB_STAT}_var-t1w4tck_warp.nii.gz

T1W_ANT_PRE=sub-${SUB_MOVE}_space_${SUB_STAT}_var-t1w_

T1W_AFF=${T1W_ANT_PRE}affine.txt
T1W_IWARP=${T1W_ANT_PRE}invwarp.nii.gz
T1W_IWARP_FIX=${T1W_ANT_PRE}InverseWarp.nii.gz

#TCK_MOVE_PRE=sub-${SUB_MOVE}
#TCK_DIR=aligned_tracts
#mkdir ${TCK_DIR}
#TCK_OUT_PRE=${TCK_DIR}/sub-${SUB_MOVE}

#------------------------------------------------------------------------------
# ANTS Registration of tractograms
#------------------------------------------------------------------------------

# Convert the warp for T1W
warpinit ${T1W_STAT} ${WARP_INI} -force -quiet
cp $T1W_IWARP $T1W_IWARP_FIX
for i in 0 1 2; do
    WarpImageMultiTransform 3 \
        ${WARP_PRE}-i${i}.nii.gz ${WARP_PRE}-t${i}.nii.gz \
        -R $T1W_STAT -i $T1W_AFF $T1W_IWARP_FIX
done
rm -f $T1W_IWARP_FIX
warpcorrect $WARP_TMP $WARP_T1W -force -quiet
rm ${WARP_PRE}-*


# Apply the warp
while read tract_name; do         
    echo "Tract name: $tract_name";
    if [ ! -d "aligned_examples_directory_$tract_name" ]; then
  	mkdir aligned_examples_directory_$tract_name;
    fi
    tract=examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.trk'
    python trk2tck.py ${tract}
    tcknormalise examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.tck' \
	$WARP_T1W aligned_examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.tck' -force -quiet
    python tck2trk.py ${T1W_MOVE} aligned_examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.tck'
    #rm examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.tck'
    #rm aligned_examples_directory_$tract_name/$id_mov'_'$tract_name'_tract.tck'
done < tract_name_list.txt



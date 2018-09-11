function main_segSuperset(stat_sub, fsDir)

if ~isdeployed
	addpath(genpath('/N/u/brlife/git/wma'));
	addpath(genpath('/N/u/brlife/git/vistasoft'));
end

tractogram = 'tractogram_static.tck';
    
classification_parctpc = wma_segSuperset_pArcTPC(tractogram, fsDir);
index=classification_parctpc.index;
save(sprintf('supersets_idx/%s_parctpc_index.mat',num2str(stat_sub)), 'index', '-v7');
    
classification_mdlf = wma_segSuperset_MdLF(tractogram, fsDir);
index=classification_mdlf.index;
save(sprintf('supersets_idx/%s_mdlf_index.mat',num2str(stat_sub)), 'index', '-v7');


exit;
end

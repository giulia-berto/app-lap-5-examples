clc;
close all;
clear all;


for sub=[910241]
    
    tractogram_dir = '/home/giulia/Desktop/HCP3-VWM-indiana/derivatives/brain-life.app-ensembletracking';
    fsDir = sprintf('/home/giulia/Downloads/%s_output',num2str(sub));
    tractogram = strcat(tractogram_dir,'/sub-',num2str(sub),'/sub-',num2str(sub),'_track.tck');
    
    classification_parctpc = wma_segSuperset_pArcTPC(tractogram, fsDir);
    index=classification_parctpc.index;
    save(sprintf('%s_parctpc_index.mat',num2str(sub)), 'index', '-v7');
    
    %classification_mdlf = wma_segSuperset_MdLF(tractogram, fsDir);
    %index=classification_mdlf.index;
    %save(sprintf('%s_mdlf_index.mat',num2str(sub)), 'index', '-v7');
    
end
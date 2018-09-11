function [classification] = wma_segSuperset_MdLF(feORwbfg, fsDir)

% Inputs:
% -wbfg: a whole brain fiber group structure (e.g. 615441_track.tck)
% -fsDir: path to THIS SUBJECT'S freesurfer directory (e.g. 615441_output)

% Outputs:
% classification:  a classification structure with .name and .indexes
% fields
%
% (C) Daniel Bullock, 2018, Indiana University


%% parameter note & initialization

[wbfg, fe] = bsc_LoadAndParseFiberStructure(feORwbfg)

for iFibers=1:length(wbfg.fibers)
    fiberNodeNum=round(length(wbfg.fibers{iFibers})/2);
    curStreamline=wbfg.fibers{iFibers};
    midpoints(iFibers,:)=curStreamline(:,fiberNodeNum);
    endpoints1(iFibers,:)=curStreamline(:,1);
    endpoints2(iFibers,:)=curStreamline(:,end);
    %streamLengths(iFibers)=sum(sqrt(sum(diff(wbfg.fibers{iFibers},1,2).^2)));
end

labelNifti= wma_getAsegFile(fsDir , '2009');
LeftStreams=midpoints(:,1)<0;

classification=[];

classification.index=zeros(length(wbfg.fibers),1);
classification.names={'Left_MdLF','Right_MdLF'};

parietalROIs3=[157, 127, 168, 136, 126, 125];
LatTempROI=[134, 144, 174, 135];


for leftright= [1,2]
    
    sidenum=10000+leftright*1000;  
       
    %merge ROIs
    [mergedParietalROI] = bsc_roiFromFSnums(fsDir,parietalROIs3+sidenum,1,5);
    mergedParietalROI.name='parietalROI';
    
    [mergedLatTempROI] = bsc_roiFromFSnums(fsDir,LatTempROI+sidenum,1,9);
    mergedLatTempROI.name='lateral-temporalROI';
    
    %set operands for ROIS
    operands={'endpoints','endpoints'};
    
    %segmentation 
    currentROIs= [{mergedParietalROI} {mergedLatTempROI}];
    [MdLF, MdLF_idx]=wma_SegmentFascicleFromConnectome(wbfg, currentROIs, operands, 'MdLF');
    
    
    if leftright==1
        fprintf('\n Left segmentation complete')
        classification.index(MdLF_idx & LeftStreams)=find( strcmp(classification.names,'Left_MdLF'));

    else
        fprintf('\n Right segmentation complete')
        classification.index(MdLF_idx & ~LeftStreams)=find( strcmp(classification.names,'Right_MdLF'));
    end
    
end

end

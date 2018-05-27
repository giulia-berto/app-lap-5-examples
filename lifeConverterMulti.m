function out = lifeConverterMulti(fe_filename, ref_src_filename)

if ~isdeployed
    disp('adding path')
    addpath(genpath('/N/u/brlife/git/vistasoft'));
    addpath(genpath('/N/u/brlife/git/jsonlab'));
    addpath(genpath('/N/u/brlife/git/o3d-code'));
    addpath(genpath('/N/u/brlife/git/encode'));
end

% Convert positively-weighted streamlines from LiFE into .trk
if ischar(fe_filename)
	fe_src = fullfile(fe_filename);
else
	fe_src = fullfile(char(fe_filename));
end 

if ischar(ref_src_filename)
	ref_src = fullfile(ref_src_filename);
else
	ref_src = fullfile(char(ref_src_filename));
end

fe2trk(fe_src, ref_src, 'output.trk');

exit;
end

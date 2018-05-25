function out = lifeConverter1(fe, ref_src)

if ~isdeployed
    disp('adding path')
    addpath(genpath('/N/u/brlife/git/vistasoft'));
    addpath(genpath('/N/u/brlife/git/jsonlab'));
    addpath(genpath('/N/u/brlife/git/o3d-code'));
    addpath(genpath('/N/u/brlife/git/encode'));
end

% Convert positively-weighted streamlines from LiFE into .trk
fe2trk(fe, ref_src, 'output.trk');

exit;
end

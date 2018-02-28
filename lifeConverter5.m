function out = lifeConverter5()

addpath(genpath('/N/u/hayashis/BigRed2/git/vistasoft'));
addpath(genpath('/N/u/brlife/git/jsonlab'));
addpath(genpath('/N/u/brlife/git/o3d-code'));
addpath(genpath('/N/u/brlife/git/encode'));

config = loadjson('config.json');

fe_src_static = fullfile(config.tractogram_static);
ref_src_static = fullfile(config.t1_static);
trk_out_static = 'life_static_output.trk';

fe_src_moving1 = fullfile(config.tractogram_moving1);
ref_src_moving1 = fullfile(config.t1_moving1);
trk_out_moving1 = 'life_moving_output1.trk';

fe_src_moving2 = fullfile(config.tractogram_moving2);
ref_src_moving2 = fullfile(config.t1_moving2);
trk_out_moving2 = 'life_moving_output2.trk';

fe_src_moving3 = fullfile(config.tractogram_moving3);
ref_src_moving3 = fullfile(config.t1_moving3);
trk_out_moving3 = 'life_moving_output3.trk';

fe_src_moving4 = fullfile(config.tractogram_moving4);
ref_src_moving4 = fullfile(config.t1_moving4);
trk_out_moving4 = 'life_moving_output4.trk';

fe_src_moving5 = fullfile(config.tractogram_moving5);
ref_src_moving5 = fullfile(config.t1_moving5);
trk_out_moving5 = 'life_moving_output5.trk';


% Convert positively-weighted streamlines from LiFE into .trk
fe2trk(fe_src_static, ref_src_static, trk_out_static);
fe2trk(fe_src_moving1, ref_src_moving1, trk_out_moving1);
fe2trk(fe_src_moving2, ref_src_moving2, trk_out_moving2);
fe2trk(fe_src_moving3, ref_src_moving3, trk_out_moving3);
fe2trk(fe_src_moving4, ref_src_moving4, trk_out_moving4);
fe2trk(fe_src_moving5, ref_src_moving5, trk_out_moving5);


% Create tractograms directory
mkdir ./tractograms_directory
movefile life_moving_output1.trk ./tractograms_directory/
movefile life_moving_output2.trk ./tractograms_directory/
movefile life_moving_output3.trk ./tractograms_directory/
movefile life_moving_output4.trk ./tractograms_directory/
movefile life_moving_output5.trk ./tractograms_directory/


% Convert segmentation
load(fullfile(config.segmentation1));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving1,sprintf('%s_tract%s.trk',strrep(fg_classified(config.tract).name,' ','_'),'1'));
load(fullfile(config.segmentation2));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving2,sprintf('%s_tract%s.trk',strrep(fg_classified(config.tract).name,' ','_'),'2'));
load(fullfile(config.segmentation3));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving3,sprintf('%s_tract%s.trk',strrep(fg_classified(config.tract).name,' ','_'),'3'));
load(fullfile(config.segmentation4));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving4,sprintf('%s_tract%s.trk',strrep(fg_classified(config.tract).name,' ','_'),'4'));
load(fullfile(config.segmentation5));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving5,sprintf('%s_tract%s.trk',strrep(fg_classified(config.tract).name,' ','_'),'5'));


% Create examples directory
mkdir ./examples_directory;
movefile *tract1.trk ./examples_directory/
movefile *tract2.trk ./examples_directory/
movefile *tract3.trk ./examples_directory/
movefile *tract4.trk ./examples_directory/
movefile *tract5.trk ./examples_directory/


% tmp file to extract the tract name
load(fullfile(config.segmentation1));
write_fg_to_trk(fg_classified(config.tract),ref_src_moving1,sprintf('%s_tract.trk',strrep(fg_classified(config.tract).name,' ','_')));


exit;
end

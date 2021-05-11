% %% load simulated data
% data = struct;
% path = '../python/data/';
% filename = 'sim_data_cs_4um8um_shift-50.mat';
% load(strcat(path, filename),'data_final','data_sep_final','param_final');
% data_raw = data_final';
% labels = repelem(1:4, size(data_raw,2)/4);
% % labels = ones(1, size(data_raw,2));
% data.n_features = length(data_sep_final{1});
% opt.data_type = 'sim';
% opt.ds = 1;
% 
% % select desired template
% variableRange = [14,20,26,32,38];
% str = string((700:100:1300)');
% % str = string((800:200:1200)');
% tmplt_r2 = cell(length(str),1); tmplt_r4 = cell(length(str),1); 
% min_idx = zeros(length(str),1); max_idx = zeros(size(min_idx));
% dec = 100;
% 
% for cnt = 1:length(str)
%     
%     % read in data from 2um radius
%     filename = strcat(path,'signals/complete_complete_1.5-2.5_',str(cnt),'_-50 - 50_20E-9.txt');
%     opts = detectImportOptions(filename, 'DecimalSeparator', ',');
%     opts.SelectedVariableNames = variableRange;
%     tmplt_r2{cnt} = readmatrix(filename, opts);
%     
%     % read in data from 4um radius
%     filename = strcat(path,'signals/complete_complete_3-5_',str(cnt),'_-50 - 50_20E-9.txt');
%     opts = detectImportOptions(filename, 'DecimalSeparator', ',');
%     opts.SelectedVariableNames = variableRange;
%     tmplt_r4{cnt} = readmatrix(filename, opts);
%     
%     % discard values that are too small
%     min_idx(cnt) = min(find(tmplt_r2{cnt}(:,end) > max(abs(tmplt_r2{cnt}(:,end)))/dec, 1, 'first'), ...
%         find(tmplt_r4{cnt} > max(abs(tmplt_r4{cnt}))/dec, 1, 'first'));
%     max_idx(cnt) = max(find(tmplt_r2{cnt}(:,end) < -max(abs(tmplt_r2{cnt}(:,end)))/dec, 1, 'last'), ...
%         find(tmplt_r4{cnt}(:,end) < -max(abs(tmplt_r4{cnt}(:,end)))/dec, 1, 'last'));
%     
% end
% 
% tmplt = cell(1, sum(cellfun(@(x)size(x,2), {tmplt_r2; tmplt_r4})));
% cnt_3 = 1;
% for cnt = 1:length(tmplt_r2)
%     for cnt_2 = 1:size(tmplt_r2{cnt},2)
%         tmplt{cnt_3} = tmplt_r2{cnt}(min_idx(cnt):max_idx(cnt),cnt_2);
%         tmplt{cnt_3+1} = tmplt_r4{cnt}(min_idx(cnt):max_idx(cnt),cnt_2);
%         cnt_3 = cnt_3 + 2;
%     end
% end
% tmplt = cellfun(@(x) x(1:opt.ds:end), tmplt, 'UniformOutput', false);
% % tmplt = [tmplt_r2{1}(min_idx:max_idx), tmplt_r4{1}(min_idx:max_idx)];

%% load experimental data

% load input data
data = struct;

% load('../python/data/exp_data_4um8um.mat')
% data_raw = [data_4um; data_8um; data_4um8um]';
% data_raw = data_raw - mean(data_raw);
% labels = [labels_4um, labels_8um, labels_4um8um];

% load('../python/data/exp_data_train.mat')
% data_raw = [data_4um_train; data_8um_train; data_4um8um_train]';
% sz_diff = 2000 - size(data_raw,1);
% data_raw = data_raw - mean(data_raw);
% data_raw = [zeros(floor(sz_diff/2), size(data_raw,2)); data_raw; zeros(ceil(sz_diff/2), size(data_raw,2))];
% labels = ones(1, size(data_raw,2));

load('../python/data/exp_data_eval.mat')
data_raw = [data_4um_eval; data_8um_eval; data_4um8um_eval]';
data_raw = data_raw - mean(data_raw);
labels = [labels_4um_eval, labels_8um_eval, labels_4um8um_eval];

data.n_features = size(data_raw,1);
data_sep_final = [];
opt.data_type = 'exp';

% load templates for sensing matrix
load('tmplts_exp.mat')
tmplt = t_095(1:opt.ds:end,:);

%%
disp('Loading completed.')

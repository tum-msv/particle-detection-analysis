%% Script to create a dataset for deep learning approach for cell analysis
% The dataset comprises cases from 1 to 4 particles each featuring a predefined
% amount of samples. The samples vary in their diameter, speed, amplitude and
% delay of the particles and are saved as a 5D-matrix. From this matrix
% samples are drawn without replacement to create the data for each class.
% The corresponding class labels (amounts), speeds and diametres are
% stored as well for training of the proceeding algorithm.

path_raw = "../python/data/signals/";
filename_pre = "complete_complete_";
filename_suf = "-50 - 50_20E-9.txt";

%% parameters for dataset
seed = 999999999;% 1234,'default', 999999999
rng(seed);
n_features = 2000; % supposed number of samples in one observation
rad_use = 11:41; %2:50; particles within 3 times standard deviation
delr_var = 0.01; % radius variation fraction step for same particles
shift_grid = -300:50:300; % grid for shifted versions
amp_bnd = [0.01, 0.2; 0.2, 0.5];
amp_mean = mean(amp_bnd,2);
amp_std = [0.03; 0.05];
diff_amp_bnd = diff(amp_bnd,1,2);
n_labels = 4; % number of labels
samp_label = 250; % amount of data samples per label
sig_delr = (length(rad_use)-1)/2/3;
pdf_delr = @(x) 1/(sqrt(2*pi)*sig_delr) * exp(-1/2*(x./sig_delr).^2);
mu_delr_idx = (length(rad_use)-1)/2;
weights_delr = pdf_delr(-mu_delr_idx:mu_delr_idx);
% weights_delr = ones(1, length(rad_use));
weights_delr = weights_delr / sum(weights_delr);
path_save = "../python/data/";
filename_save = "data.mat";

%% read in of data
r_str = ["1.5-2.5"; "3-5"];
v_str = ["_700_"; "_800_"; "_900_"; "_1000_"; "_1100_"; "_1200_"; "_1300_"];
data_use_all = zeros(length(r_str), length(v_str), length(shift_grid), n_features, length(rad_use));

for i = 1:length(r_str)
    
    for j = 1:length(v_str)
        
        complete_path = strcat(path_raw, filename_pre, r_str(i), v_str(j), filename_suf);
        data = readmatrix(complete_path, 'DecimalSeparator', ',');
        
        % only use columns specified in parameters
        data_use = data(:,rad_use);
        
        % repeat first and last value until size M is reached (l_diff > 0)
        l_diff = n_features - size(data_use,1);
        pad_pre = floor(l_diff/2);
        pad_post = ceil(l_diff/2);
        data_use = padarray(data_use, pad_pre, 'replicate', 'pre');
        data_use = padarray(data_use, pad_post, 'replicate', 'post');
        
        % store shifted versions in array
        for k = 1:length(shift_grid)
            data_use_all(i,j,k,:,:) = circshift(data_use, shift_grid(k));
        end
        
    end
    
end
disp("Data read in.")

%% creation of multi-particle cases with labeling
data_final = zeros(n_labels*samp_label, n_features);
data_sep_final = cell(n_labels*samp_label,1);
label_final = zeros(n_labels*samp_label, 1);
param_final = zeros(n_labels*samp_label, 3*n_labels); % diameter, diameter offset, velocity

for i = 1:n_labels
    
    % draw samples from radius and speed
    r = randi(length(r_str), samp_label, i);
    v = randi(length(v_str), samp_label, i);
    
    % draw samples from amplitude for amount of particles
    amp = zeros(samp_label, i);
    k = 1;
    while k <= i
        amp(:,k) = amp_mean(r(:,k),1) + randn(samp_label,1) .* amp_std(r(:,k));
%         amp(:,k) = amp_bnd(r(:,k),1) + rand(samp_label,1) .* (amp_bnd(r(:,k),2)-amp_bnd(r(:,k),1));
        if all(amp(:,k) > 0), k = k + 1; end
    end
    
    for j = 1:samp_label
        
        % set element counter
        cnt = (i-1)*samp_label+j;
        
        % sample shift and delta_r
        shift = sort(randperm(length(shift_grid), i), 'ascend');
        delr = datasample(1:length(rad_use), i, 'Weights', weights_delr, 'Replace', false);
        
        % draw sample
        data_tmp = zeros(n_features,i);
        for l = 1:i
            data_tmp(:,l) = squeeze(data_use_all(r(j,l),v(j,l),shift(l),:,delr(l)));
        end
        
        % scale with amplitude
        data_tmp = data_tmp ./ max(abs(data_tmp)) .* amp(j,:);
        
        % save sample and label
        data_sep_final{cnt} = data_tmp';
        data_final(cnt,:) = sum(data_tmp,2)';
        label_final(cnt) = i;
        
        % save true (exact) cell parameters for all particles
        for p = 1:i
            
            % true radius from namestring
            r_true = mean(str2double(strsplit(r_str(r(j,p)), '-')));
            
            % radius offset from sampled offset
            delr_true = (rad_use(delr(p)) - rad_use(mu_delr_idx+1)) * r_true * delr_var;
            
            % true velocity from namesting
            v_true = str2double(erase(v_str(v(j,p)), '_'));
            
            % store parameters of this particles
            param_final(cnt,(p-1)*3+1:p*3) = [r_true, delr_true, v_true];
            
        end
        
    end
    
end
disp("Dataset created.")

%% save generated data
save(strcat(path_save,filename_save), 'data_final', 'param_final', 'data_sep_final')
disp("Dataset saved.")

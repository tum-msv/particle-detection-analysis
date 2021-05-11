%% options for CS
relCorr1 = []; relCorr2 = []; relCorr3 = []; relCorr4 = [];
rng('default');

opt.L = 2; % sparsity level
opt.n_features = (2000)/opt.ds; % number of samples M in observation
opt.gap = (600)/opt.ds; % same as opt.border
opt.csextract = false; % flag to say if data should be extracted
opt.flagplot = false; % flag to plot reconstruction level comparison
opt.flagawgn = false; % flag to add additional noise
opt.flageval = 0; % flag to evaluate CS
opt.verbose = 1; % verbose option
opt.algo = 'AdOMP'; % string which algorithm to use for CS
opt.method = 'single_lm'; % joint optimisation to use
opt.grid_elno = 0.1*opt.n_features; % grid of delays
opt.smat = 'bsone'; % version of sensing matrix to be used

rangeSNRdb = 30;%-5:1:30; % SNR range
loopsSNR = length(rangeSNRdb);

%% execution
results_file = 'results';
fID = fopen(strcat('results/',results_file,'.txt'), 'a');
% rangeSpec = [labels==1; labels==2; labels==3; labels==4];
rangeSpec = labels == 1;
% rangeSpec = 1:10:size(data_raw,2);

results = cell(loopsSNR, 4);
for i = 1:loopsSNR
    
    for j = 1:size(rangeSpec,1)
        
        res = struct;
        
        fprintf(fID, 'Loop %d of %d @ SNR=%d.\n', j, size(rangeSpec,1), rangeSNRdb(i));
        data_raw_use = data_raw(1:opt.ds:end,rangeSpec(j,:));
        opt.snr = 10^(rangeSNRdb(i)/10); % snr for awgn
        p_signals = mean(data_raw_use.^2,'all');
        data.obs = data_raw_use + sqrt(p_signals/opt.snr) * randn(size(data_raw_use));
        if ~isempty(data_sep_final)
            data.obs_true = data_sep_final(rangeSpec(j,:));
        else
            data.obs_true = [];
        end
        data.labels = labels(rangeSpec(j,:));
        % CSmodel = CS(data, generator.coeff_gauss, tmplt, opt);
        CSmodel = struct('type', opt.data_type);
        [CSmodel.lvlMse, CSmodel.lvlCrl, CSmodel.beads, CSmodel.y_rec, CSmodel.sup, CSmodel.err, CSmodel.prm] = ...
            processDataWithCS(data.obs, data.labels, tmplt, opt.grid_elno, opt.smat, opt.algo, ...
            opt.flageval, data.obs_true, opt.method, opt.flagplot, opt.verbose);
        try
            CSmodel.prm_conv = convertParameters(CSmodel.prm);
        catch e
            disp(e)
        end
        
        % evaluate simulated data (true reconstructions are available)
        if strcmp(opt.data_type, 'sim')
            
            % calculate error statistics
            CSmodel.mse = mean(CSmodel.lvlMse, 'all', 'omitnan');
            CSmodel.sig = std(CSmodel.lvlMse, [], 'all', 'omitnan');
            CSmodel.med = median(CSmodel.lvlMse, 'all', 'omitnan');
            CSmodel.mav = max(CSmodel.lvlMse(:));
            fprintf(fID, 'MSE: %.2E, \t Std: %.2E, \t Med: %.2E, \t Max: %.2E.\n\n', ...
                CSmodel.mse, CSmodel.sig, CSmodel.med, CSmodel.mav);
            
            % save error in results
            res.err = CSmodel.lvlMse;
            
        elseif strcmp(opt.data_type, 'exp')
            
            % calculate error statistics
            CSmodel.mse = mean(CSmodel.err);
            CSmodel.sig = std(CSmodel.err);
            CSmodel.med = median(CSmodel.err);
            CSmodel.mav = max(CSmodel.err);
            fprintf(fID, 'MSE: %.2E, \t Std: %.2E, \t Med: %.2E, \t Max: %.2E.\n\n', ...
                CSmodel.mse, CSmodel.sig, CSmodel.med, CSmodel.mav);
            
            % save total error in results
            res.err = CSmodel.err;
            
        end
        
        % save parameters
        res.prm = CSmodel.prm;
        res.prm_conv = CSmodel.prm_conv;
        results{i,j} = res;
        save(strcat('results/',results_file,'.mat'), 'results')
        
    end
    
end

fclose(fID);
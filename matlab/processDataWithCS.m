function [allMse, allCrl, beads, y_lvl_arr, allSup, err, allPrm] = ...
    processDataWithCS(data, labels, f, n_gridpts, smat, algo, flagEval, data_eval, method, flagPlot, verbose)
%processDataWithCS Processes input data with Compressive Sensing
% Input:
% data:             data to process (one sample in every column)
% labels:           number of cells/particles in every data column
% f:                template(s) for sensing matrix
% n_gridpts:        number of elements in the grid for the sensing matrix
% smat:             version of sensing matrix to use
% algo:             algorithm to use for CS ('NOMP' or 'AdOMP')
% flagEval:         flag to specify whether evaluation data are available or not
% data_eval:        cell array which contains all true separated signals
% method:           string which AdOMP optimisation technique to use (use 'single_lm')
% flagPlot:         flag to plot comparison of reconstruction
% verbose:          verbose option
% Output:
% allMse:           mse of all reconstruction (NaN for incorrect amount or unavailable)
% allCrl:           corrleation of all reconstructions (same as allMSE)
% beads:            true and found amount of particles (only relevant if CS is used for detection)
% y_lvl_arr:        cell array that contains all reconstructions
% allSup:           array that contains all found supports
% err:              array that contains all CS errors
% allPrm:           array that contains all cell parameters

% create delay grid
if iscell(f)
    Nf = length(f{end});
else
    Nf = size(f, 1);
end
[n_features, n_data] = size(data);
G_max = n_features - Nf;
G = floor(linspace(0, G_max, n_gridpts));

% create sensing matrix
Phi = createSensingMatrix(G, n_features, n_gridpts, f, smat);

% assign output arrays
maxAmount = max(labels);
allSup = zeros(n_data, maxAmount);
err = zeros(n_data, 1);
y_lvl_arr = cell(n_data,1);
allMse = NaN(n_data, maxAmount);
allCrl = zeros(n_data, maxAmount);
beads = zeros(n_data, 2);
allPrm = cell(n_data, 1);

% loop over all data samples
for cnt = 1:n_data
    
    % get input data for algorithm
    y = data(:,cnt);
    trueAmount = labels(cnt);
    prm_rec = struct;
    
    % apply CS
    if strcmp(algo, 'NOMP')
        [y_rec, sup, rN] = NOMP(y, Phi, trueAmount);
    elseif strcmp(algo, 'AdOMP')
        [y_rec, sup, prm_rec, rN] = AdOMP(y, Phi, trueAmount, method);
    else
        error('Specify correct solution algorithm for CS!');
    end
    
    % get gauss parameters if choice is not AdOMP
    if ~strcmp(algo, 'AdOMP')
        [~, prm_rec] = fitDerivGauss(y_rec);
        prm_rec = rmfield(prm_rec, {'lb','ub'});
    end
    
    % evaluate
    if flagEval
        
        % true separated signals are available
        y_true = data_eval{cnt};
        if size(y_true,1) ~= size(y_rec,1)
            y_true = y_true';
        end
        [y_rec, allMse(cnt,:), allCrl(cnt,:), beads(cnt,:), prm_rec] = ...
            evalCS(rN, y_true, y_rec, sup, trueAmount, maxAmount, prm_rec);
        
    else
        
        % only extract parameters of reconstructed signals
        [prm_rec.magD, prm_rec.normInt] = extract_params_rec(y_rec, prm_rec);
        beads(cnt,:) = [trueAmount, size(y_rec,2)];
        y_true = [];
        
    end
    
    % plot if flag is set
    if flagPlot
        fprintf('Total error @ sample %d (with noise):', cnt)
        disp(rN)
        fprintf('Per level error:')
        disp(allMse(cnt, 1:beads(cnt,2)))
        fprintf('\n Per level correlation:')
        disp(allCrl(cnt, 1:beads(cnt,2)))
        plotComp(y_true, y_rec, y, beads(cnt,1), beads(cnt,2));
    end
    
    % store support, error and reconstructed signals
    allSup(cnt, 1:length(sup)) = sup;
    err(cnt) = rN;
    y_lvl_arr{cnt} = y_rec;
    
    % order parameters in same way as reconstructed signals
    allPrm{cnt} = prm_rec;
    
    if and(~mod(cnt, 100), verbose > 0)
        fprintf('CS for %.0d patterns done.\n', cnt)
    elseif verbose > 1
        fprintf('CS for %.0d patterns done.\n', cnt)
    end
    
end

end

function [y_rec, lvlMse, lvlCrl, beads, prm_rec] = ...
    evalCS(rN, y_true, y_rec, sup, trueAmount, maxAmount, prm_rec)
%evalCS Evaluates errors and correct particle number
% Input:
% rN:         squared normalised residual norm
% y_true:     ground truth signal of every level
% y_rec:      reconstruction of every level
% sup:        found support of CS
% trueAmount: true number of particles in sample
% maxAmount:  maximum possible amount of particles
% prm_rec:    fitted parameters of super-gauss (optional)
% Output:
% y_rec:      reconstructions assigned to the correct signal
% lvlMse:     mse of every reconstruction level
% lvlCrl:     correlation of every reconstruction level
% beads:      true and found amount in column vector
% prm_rec:    sorted parameters of reconstruction

% find hidden sparsity level
if rN == Inf
    foundAmount = 0;
else
    foundAmount = length(sup);
end

n_events = size(y_rec,2);

% calculate norms of reconstructions and true signals
yN = sqrt(sum(y_true.^2,1));
yN_rec = sqrt(sum(y_rec.^2,1));
err_all = zeros(foundAmount, trueAmount);
crl_all = zeros(foundAmount, trueAmount);

% save magnetic diameter and normalised integral of reconstructions
[magD, normInt] = extract_params_rec(y_rec, prm_rec);

for i = 1:n_events
    
    % calculate reconstruction errors and correlations of all signal with regard to all true signals
    for j = 1:trueAmount
        err_all(i,j) = sum(((y_rec(:,i) - y_true(:,j))) .^ 2 / (yN(j)*yN_rec(i)));
        crl_all(i,j) = sum((y_rec(:,i) .* y_true(:,j)) / (yN(j)*yN_rec(i)));
    end
    
end

pairs = zeros(foundAmount, 2);
err_all_save = err_all;

for i = 1:n_events
    
    % find minimum error and save pairs
    [idx_rec, idx_true] = find(err_all == min(min(err_all)));
    pairs(i,:) = [idx_rec(1), idx_true(1)];
    err_all(idx_rec, :) = Inf; 
    err_all(:, idx_true) = Inf;
    
end

err_all = err_all_save;

% sort in order of true signal
pairs = sortrows(pairs, 2);

% save error and correlation if sparsity level correct
if trueAmount == foundAmount
    y_rec = y_rec(:,1:foundAmount);
    
    % sort reconstructed signals and corresponding parameters in the found order
    y_rec = y_rec(:, pairs(:,1));
    magD = magD(pairs(:,1));
    normInt = normInt(pairs(:,1));
    
    % save per level mse
    idx_lin = sub2ind(size(err_all), pairs(:,1), pairs(:,2));
    lvlMse = err_all(idx_lin)';
    lvlMse(foundAmount+1:maxAmount) = NaN;
    
    % calculate correlation
    lvlCrl = crl_all(idx_lin)';
    lvlCrl(foundAmount+1:maxAmount) = NaN;
    
else
    
    % otherwise set to NaN
    lvlMse = NaN(1, maxAmount);
    lvlCrl = NaN(1, maxAmount);
    
end

% save true value and found one
beads = [trueAmount, foundAmount];

% sort remaining parameters in found order and append mag_D and norm_Int
prm_rec = structfun(@(x) x(pairs(:,1)), prm_rec, 'UniformOutput', false);
prm_rec.magD = magD;
prm_rec.normInt = normInt;

end
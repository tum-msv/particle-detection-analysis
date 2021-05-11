function [mag_D, norm_int] = extract_params_rec(y_rec, prm_rec)
%extract_params_rec Extracts parameters from cell signal reconstructions
% Input:
% y_rec:        reconstructed signals
% prm_rec:      super-gauss parameters of the reconstructions
% Output:
% mag_D:        magnetic diameters of reconstructions
% norm_int:     normalised integrals of reconstructions

n_rec = size(y_rec,2);
mag_D = zeros(n_rec,1);
norm_int = zeros(n_rec,1);

for i = 1:n_rec
    
    % select current reconstructions
    y = y_rec(:,i);
    
    % find magnetic diameter
    if nargin < 2
        
        % only use recovered signal
        [~, minLoc] = findpeaks(-y, 'SortStr', 'descend');
        [~, maxLoc] = findpeaks( y, 'SortStr', 'descend');
        mag_D(i) = mean(abs(maxLoc - minLoc));
        
    else
        
        % use fitted parameters (if available)
        p = struct('mu', prm_rec.mu1(i), 'sigma', prm_rec.sigma(i), 'p1', prm_rec.p1(i), 'p2', prm_rec.p2(i));
        x = (1:length(y))';
        y_half = sign(p.mu-x) .* abs(p.mu-x).^p.p2/p.sigma^2 .* exp(-abs(x-p.mu).^p.p1 / (2*p.sigma^2));
        [~, minIdx_half] = min(y_half);
        [~, maxIdx_half] = max(y_half);
        mag_D(i) = abs(minIdx_half - maxIdx_half);
        
    end
    
    % calculate normalized integral
    norm_int(i) = trapz(abs(-1 + 2*(y-min(y)) / (max(y)-min(y))));
    
end

end
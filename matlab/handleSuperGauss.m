function y_fit = handleSuperGauss(x, g_sup, n_pts, n_events)
% function y_fit = handleSuperGauss(x, g_sup, n_pts, n_events)
%
% Defines function handle for a superposition of derivatives of gaussians.
%
% Inputs:
% x:            parameter vector containing all means, variances etc.
% g_sup:        function handle for evaluation
% n_pts:        number of points
% n_events:     number of events
% Outputs:
% y_fit:        evaluations of dgaussians

% calculate size and evaluation array
y_fit = zeros(n_pts, n_events);
n_prm = length(x) / n_events;

for i = 1:n_events
    
    % evaluate for every value
    y_fit(:,i) = g_sup(x((i-1)*n_prm+1:i*n_prm));
    
end

end
function [y_g, prm_fit] = fitDerivGauss(y_rec)
% function [y_g, prm] = fitDerivGauss(y_rec, r)
%
% Fit a superposition of derivatives of gaussian functions to the residual.
%
% Inputs
% y_rec:        reconstructed version of residual
% Outputs
% y_g:          fit of r with superposition of derivatives of gaussians
% prm_fit:      struct which contains fitted parameters

% handle for derivative of gaussian
G = @(x, mu, sigma, p1) exp(-abs(x - mu).^p1 / (2*sigma^2));
g = @(x, mu, sigma, p1, p2) sign(mu - x) .* abs(mu - x).^p2 / sigma^2 .* G(x, mu, sigma, p1);

% handle for superposition of preceding
[n_pts, n_events] = size(y_rec);
g_sup = @(pts, mu1, mu2, sigma, a, p1, p2) a * (g(pts, mu1, sigma, p1, p2) + g(pts, mu2, sigma, p1, p2));

% set preliminaries
y_g = zeros(n_pts, n_events);
pts = (1:n_pts)';
n_prm = 6;
mu1 = zeros(n_events,1);
mu2 = zeros(n_events,1);
sigma = zeros(n_events,1);
a = zeros(n_events,1);
p1 = zeros(n_events,1);
p2 = zeros(n_events,1);
lb_all = zeros(n_prm*n_events,1);
ub_all = zeros(n_prm*n_events,1);

% wrapper function for fit
g_wrap = @(x) g_sup(pts, x(1), x(2), x(3), x(4), x(5), x(6)); 

for i = 1:n_events
    
    % parameters for template
    prm_y = getParamsTmplt(y_rec(:,i));
    
    % initial values for current fit
    mu10 = prm_y.zero(1);
    mu20 = prm_y.zero(3);
    sigma0 = prm_y.zero(1) - prm_y.peaksMax(1);
    a0 = max(y_rec(:,i)) / g(mu10-sigma0, mu10, sigma0, 1, 2);
    p10 = 1;
    p20 = 1;
    x0 = [mu10; mu20; sigma0; a0; p10; p20];
    
    % lower and upper bounds for current fit (determined heuristically)
    lb = [prm_y.peaksMax(1); prm_y.peaksMax(2); 1.16; 0; 0.7; 1.11];
    ub = [prm_y.peaksMin(1); prm_y.peaksMin(2); 3.68;  Inf; 1.11; 1.77];
    
    % objective function
    yN = norm(y_rec(:,i));
    fun = @(x) 1/yN * (y_rec(:,i) - g_wrap(x));
    
    % apply lsqnonlin
    opts = optimoptions(@lsqnonlin, 'FunctionTolerance', 1e-4, 'Display', 'off');
    problem = createOptimProblem('lsqnonlin', 'objective', fun, 'x0', x0, 'lb', lb, 'ub', ub, 'options', opts);
    [x, optval] = lsqnonlin(problem);
    mu1(i) = x(1); mu2(i) = x(2); sigma(i) = x(3); a(i) = x(4); p1(i) = x(5); p2(i) = x(6);
    
    % save fitted curve
    y_g(:,i) = g_wrap(x);
    
    % save lower and upper bound in array
    lb_all((i-1)*n_prm+1:i*n_prm) = lb;
    ub_all((i-1)*n_prm+1:i*n_prm) = ub;
    
end

% save fitting parameters in struct
prm_fit = struct('mu1', mu1, 'mu2', mu2, 'sigma', sigma, 'a', a, 'p1', p1, 'p2', p2, 'lb', lb_all, 'ub', ub_all);

end


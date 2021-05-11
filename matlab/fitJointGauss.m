function [y_joint, prm_joint, optval] = fitJointGauss(y, prm, method)
% function [y_g, prm] = fitDerivGauss(y_rec, r)
%
% Fit a superposition of derivatives of gaussian functions to the residual.
%
% Inputs
% y_rec:        reconstructed version of residual
% prm:          parameters from the initial gauss fit
% n_grid:       number of grid elements
% Outputs
% y_g:          jointly optimised gaussian fit to match residual
% prm_joint:    struct which contains jointly fitted parameters
% optval:       resulting optimization objective

% handle for derivative of gaussian
G = @(x, mu, sigma, p1) exp(-abs(x - mu).^p1 / (2*sigma^2));
g = @(x, mu, sigma, p1, p2) sign(mu - x) .* abs(mu - x).^p2 / sigma^2 .* G(x, mu, sigma, p1);

% handle for superposition of preceding
n_events = length(prm.mu1);
n_pts = length(y);
n_prm = 6;
pts = (1:n_pts)';
g_sup = @(x) x(4) * (g(pts, x(1), x(3), x(5), x(6)) + g(pts, x(2), x(3), x(5), x(6)));

% define objective function
yN = norm(y);
g_sup_all = @(x) handleSuperGauss(x, g_sup, n_pts, n_events);
fun = @(x) 1/yN * (y - sum(g_sup_all(x),2));

% lower and upper bound
lb = prm.lb; % [mu1_1, mu2_2, sigma_1, a_1, p1_1, p2_2]
ub = prm.ub;

% make bounds somewhat tight around found parameters
mu_diff = prm.mu2(:) - prm.mu1(:);
mu_diff_bnd = [100, 220];
lb(1:n_prm:end) = prm.mu1(:) - (mu_diff_bnd(2) - mu_diff)/2;
lb(2:n_prm:end) = prm.mu2(:) - (mu_diff - mu_diff_bnd(1))/2;
ub(1:n_prm:end) = prm.mu1(:) + (mu_diff - mu_diff_bnd(1))/2;
ub(2:n_prm:end) = prm.mu2(:) + (mu_diff_bnd(2) - mu_diff)/2;

% initial value
x0 = [prm.mu1(:), prm.mu2(:), prm.sigma(:), prm.a(:), prm.p1(:), prm.p2(:)]';
x0 = x0(:);

if strcmp(method, 'single_tr')    
    % use single nonlinear least squares run with trust-region
    opts = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', 'FunctionTolerance', 1e-4, 'Display', 'off');
    problem = createOptimProblem('lsqnonlin', 'objective', fun, 'x0', x0, 'lb', lb, 'ub', ub, 'options', opts);
    [x, optval] = lsqnonlin(problem);
    
elseif strcmp(method, 'single_lm')
    % use single nonlinear least squares run with levenberg-marquardt
    opts = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', 'FunctionTolerance', 1e-4, 'Display', 'off');
    problem = createOptimProblem('lsqnonlin', 'objective', fun, 'x0', x0, 'options', opts);
    [x, optval] = lsqnonlin(problem);
    
elseif strcmp(method, 'multi')
    % use multi search
    ms = MultiStart('FunctionTolerance', 1e-4, 'UseParallel', true, 'StartPointsToRun', 'bounds');
    opts = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', 'FunctionTolerance', 1e-4, 'Display', 'off');
    problem = createOptimProblem('lsqnonlin', 'objective', fun, 'x0', x0, 'lb', lb, 'ub', ub, 'options', opts);
    [x, optval] = run(ms, problem, 10);
    
elseif strcmp(method, 'global')
    % use global search
    gs = GlobalSearch('FunctionTolerance', 1e-4);
    opts = optimoptions(@fmincon, 'FunctionTolerance', 1e-4, 'Display', 'off');
    problem = createOptimProblem('fmincon', 'objective', @(x)norm(fun(x))^2, 'x0', x0, 'lb', lb, 'ub', ub, 'options', opts);
    [x, optval] = run(gs, problem);
    
else
    error('Specify correct solution method!')
    
end

% assign output
y_joint = g_sup_all(x);
prm_joint = struct('mu1', x(1:n_prm:end), 'mu2', x(2:n_prm:end), 'sigma', x(3:n_prm:end), ...
    'a', x(4:n_prm:end), 'p1', x(5:n_prm:end), 'p2', x(6:n_prm:end));

end

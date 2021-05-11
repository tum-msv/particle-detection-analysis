function [t, coeff] = kernel_reg(p, n_coeff, sig, lambda)
%kernelReg Calculates template with least squares kernel regression
% Input:
% p:        cell array containing patterns for kernel regression
% n_coeff:  number of coefficients for kernel regression
% sig:      standard deviation for kernel regression
% lambda:   regularisation parameter for pseudo-inverse
% Output:
% t:        template
% coeff:    coefficients belonging to template

% initialise output
n_tmplts = length(p);
Np = size(p{1},2);
t = zeros(Np, n_tmplts);
coeff = zeros(n_coeff, n_tmplts);

% define Gaussian kernel matrix
K_single = createGaussMatrix(Np, n_coeff, sig);

% iterate over all pattern collection
for i = 1:n_tmplts
    
    % define left side of equation (all samples in vector)
    P = size(p{i},1);
    F = p{i}.';
    F = F(:);
    
    % solve least squares problem
    K = kron(ones(P,1), K_single);
    coeff(:,i) = (K' * K + lambda*eye(size(K,2))) \ K' * F;
    
    % calculate kernel/template (as row vector)
    t(:,i) = K_single * coeff(:,i);
    t(:,i) = (t(:,i) - mean(t(:,i))) / norm(t(:,i));
    
end

end

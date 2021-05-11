function [y_rec, sup, prm_rec, rN] = AdOMP(y, A, max_iter, method)
%AdOMP Adaptive Orthogonal Matching Pursuit (AdOMP)
% Input:
% y:            vector of M measurements
% A:            MxN sensing matrix
% max_iter:     number of non-zero elements of sparse signal vector
% method:       which method to use for optimisation
% Output:
% y_rec:        reconstructed sparse signal
% sup:          support of x
% prm_rec:      struct containing the jointly fitted parameters
% rN:           norm of residuals

% preliminaries
y = y(:);
r = y;
sup = zeros(1,max_iter);
k = 0;

while (k < max_iter)
    k = k + 1;
    
    % correlation of residual and columns of A
    g = A'*r;
    
    % exclude already chosen support and negative correlations
    g(sup(1:k-1)) = 0;
    g(g<0) = 0;
    
    % block index of A with largest normalized correlation to g
    [~, j] = max(abs(g));
    
    % support update
    sup(k) = j;
    
    % select columns of A corresponding to support
    A_b = A(:,sup(1:k));
    if issparse(A_b); A_b = full(A_b); end
    
    % calculate pseudoinverse of selected sensing matrix
    A_b_pinv = (A_b' * A_b) \ (A_b');
    
    % residual update
    x_est = A_b_pinv*y;
    r = y - A_b*x_est;
    
    % check if x contains negative entries
    x_neg = x_est < 0;
    if any(x_neg)
        disp('Negative x!')
        x_est = x_est(~x_neg);
        k = length(x_est);
        sup = [sup(~x_neg), zeros(1, max_iter-k)];
        r = y - A(:,sup(1:k))*x_est;
    end
    
end

% fit supergauss to columns of A according to support
[~, prm_g] = fitDerivGauss(A_b .* x_est');

% perform joint optimisation to fit residual
[y_rec, prm_rec, rN] = fitJointGauss(y, prm_g, method);

end

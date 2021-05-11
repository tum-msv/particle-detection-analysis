function [y_lvl, sup, rN] = NOMP(y, A, max_iter)
%NOMP Nonnegative Orthogonal Matching Pursuit (NOMP)
% Input:
% y:        vector of M measurements
% A:        MxN sensing matrix
% max_iter: maximum number of non-zero elements of sparse signal vector
% Output:
% y_lvl:    reconstructed signal per level
% sup:      block-support of x
% rN:       norm of residuals

y = y(:);
r = y; 
sup = zeros(1,max_iter);

k = 0;
while (k<max_iter)
    k = k + 1;
    
    % correlation of residual and columns of A (each column is block)
    g = A'*r;
    
    % exclude already chosen support and negative correlations
    g(sup(1:k-1)) = 0;
    g(g<0) = 0;
    
    % block index of A with largest normalized correlation to g
    [~, j] = max(abs(g));
    
    % support update
    sup(k) = j;
    
    % select blocks of A corresponding to T
    A_b = A(:, sup(1:k)); 
    if issparse(A_b); A_b = full(A_b); end
    
    % calculate regularized pseudoinverse of selected block sensing matrix
    A_b_pinv = (A_b' * A_b) \ (A_b');
    
    % residual update
    x_est = A_b_pinv*y;
    r = y - A_b*x_est; 
    rN = norm(r)^2 / norm(y)^2;
    
    % check if x contains negative entries
    x_neg = x_est < 0;
    if any(x_neg)
        disp('Negative x!')
        x_est = x_est(~x_neg);
        k = length(x_est);
        sup = [sup(~x_neg), zeros(1, max_iter-k)];
        r = y - A(:,sup(1:k))*x_est;
        rN = norm(r)^2 / norm(y)^2;
    end
    
end

% check if any x is smaller 0 and delete those
x_neg = x_est < 0;
if any(x_neg)
%     x_est(x_neg) = 0;
    disp('Negative x!')
end    

% get right sparsity level and per level estimate
lvl = k;
sup = sup(1:lvl);
A_b = A(:, sup);
y_lvl = A_b .* x_est';

end
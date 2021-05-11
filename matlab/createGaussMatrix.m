function K = createGaussMatrix(Nt, Ncoeff, sig)
%createGaussMatrix define Gaussian kernel matrix
% Input:
% Nt        number of samples in template
% Ncoeff    number of coefficients (columns) of gauss matrix
% sig       standard deviation of gauss kernels
% Output:
% K         NtxNcoeff gauss kernel matrix
t = (0:Nt-1).';
t = repmat(t, 1, Ncoeff);
t_i = linspace(0, Nt-1, Ncoeff);
t_i = repmat(t_i, Nt, 1);
K = exp(-(t - t_i).^2 / (2*sig^2));
end
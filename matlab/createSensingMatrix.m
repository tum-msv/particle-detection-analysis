function Phi = createSensingMatrix(G, n_features, n_gridpts, f, vers)
%createSensingMatrix sensing matrix generation for CS
% Input:
% G                 grid of all delays with L elements
% n_features        number of samples in template (use 'bsone')
% n_gridpts         number of observations M
% f                 template(s)
% vers              version of sensing matrix that should be created
% Output:
% Phi               sensing matrix

Nf = size(f,1); % number of elements in template
no_f = size(f,2); % number of templates
l_G = length(G); % number of grid elements
    
if strcmp(vers, 'bsone')
    % use ordinary sparsity (this is the type from the publication)
    Phi = zeros(n_features, l_G*no_f);
    
    % place template at every grid location
    for i = 1:no_f
        if iscell(f)
            Nf = length(f{i});
        end
        G = floor(linspace(0, n_features - Nf, n_gridpts));
        for j = 1:n_gridpts
            if iscell(f)
                Phi(1+G(j):G(j)+Nf, (i-1)*n_gridpts+j) = f{i};
            else
                Phi(1+G(j):G(j)+Nf, (i-1)*n_gridpts+j) = f(:,i);
            end
        end
    end
    Phi = Phi ./ sqrt(sum(Phi.^2,1));
%     Phi = Phi ./ max(abs(Phi));
    
elseif strcmp(vers, 'gauss') % kernel matrix into sensing matrix
    Ncoeff = 100;
    K = createGaussMatrix(Nf, Ncoeff, 1);
    
    % row and column vector for gauss matrix
    r = repmat(1:Nf, 1, Ncoeff*l_G) + repelem(G, Nf*Ncoeff);
    c = repelem(1:l_G*Ncoeff, Nf);
    
    % create sparse matrix with gaussians at r and c
    Phi = sparse(r, c, repmat(K(:),l_G,1), n_features, Ncoeff*l_G);
    
else
    error('Specify correct sensing matrix version!');
end
end
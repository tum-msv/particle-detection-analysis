function prm_arr = convertParameters(prm_cell)
%convertParameters Converts the cell array of parameters into mat-array
% Input:
% prm_cell:     cell array of parameters
% Output:
% prm_arr:      mat-array of parameters

n_prm = length(fieldnames(prm_cell{1}));
prm_arr = zeros(4*length(prm_cell), n_prm);

cnt = 1;
for i = 1:length(prm_cell)
    
    % get entry as column vector
    entry = [prm_cell{i}.mu1, prm_cell{i}.mu2, prm_cell{i}.sigma, prm_cell{i}.a, ...
        prm_cell{i}.p1, prm_cell{i}.p2, prm_cell{i}.magD, prm_cell{i}.normInt];
    
    % save in array and increase counter
    l = size(entry,1);
    prm_arr(cnt:cnt+l-1, :) = entry;
    cnt = cnt + l;

end

% shrink to correct size
prm_arr = prm_arr(1:cnt-1, :);

end
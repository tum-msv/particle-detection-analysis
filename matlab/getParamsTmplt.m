function params = getParamsTmplt(y, p)
%getParamsTmplt Extracts relevant parameters from particle template
% Input:
% y:        single cell signal (e.g. found with CS)
% p:        fit parameters of single Wheatstone bridge signal (optional)
% Output:
% params:   struct with fields magD (magnetic diameter), normInt
%           (normalized integral), peaksMin (min locations and values),
%           peaksMax (max locations and values)

% check if template is only zeros
if all(~y)
    disp('Template is only zeros!')
end

% find peaks
[minVal, minLoc] = findpeaks(-y, 'SortStr', 'descend');
[maxVal, maxLoc] = findpeaks( y, 'SortStr', 'descend');

% order idices and values correspondingly
if length(minLoc) > 1
    [minLoc, idx] = sort(minLoc(1:2), 'ascend');
    minVal = -minVal(idx);
end
if length(maxLoc) > 1
    [maxLoc, idx] = sort(maxLoc(1:2), 'ascend');
    maxVal = maxVal(idx);
end

% find location of zeros crossings
zero(1) = maxLoc(1) - 0.5 + find(y(maxLoc(1):minLoc(1)) < 0, 1, 'first');
try
    zero(2) = minLoc(1) - 0.5 + find(y(minLoc(1):maxLoc(2)) > 0, 1, 'first');
    zero(3) = maxLoc(2) - 0.5 + find(y(maxLoc(2):minLoc(2)) < 0, 1, 'first');
catch e
    disp(e)
    zero(2) = mean([maxLoc; minLoc]);
end

% find magnetic diameter
if nargin < 2
    % only use recovered signal
    magD = mean(abs(maxLoc - minLoc));
else
    % use fitted parameters (if available)
    x = (1:length(y))';
    y_half = sign(p.mu-x) .* abs(p.mu-x).^p.p2/p.sigma^2 .* exp(-abs(x-p.mu).^p.p1 / (2*p.sigma^2));
    [~, minIdx_half] = min(y_half);
    [~, maxIdx_half] = max(y_half);
    magD = abs(minIdx_half - maxIdx_half);
end

% calculate normalized integral
normInt = trapz(abs(-1 + 2*(y-min(y)) / (max(y)-min(y))));

% assign to struct
params = struct('peaksMin', [minLoc(:), minVal(:)], ...
                'peaksMax', [maxLoc(:), maxVal(:)], ...
                'zero',     zero, ...
                'magD',     magD, ...
                'normInt',  normInt);

end
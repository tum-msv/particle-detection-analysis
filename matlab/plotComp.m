function plotComp(y_true, y_rec, y, trueAmount, foundAmount)
%plotComp Plots a comparison of the found and true signals
% Input:
% y_true:        original signal(s) of every level
% y_rec:         reconstructed signal(s) of every level
% y:             input signal used for CS
% trueAmount:    true sparsity level
% foundAmout:    found sparsity level by algorithm

fig = figure('Position', [10, 10, 1800, 1000]); ax = [];
title(['True: ' int2str(trueAmount) ', Found: ' int2str(foundAmount)]);
if ~isempty(y_true)
    numcols = min([size(y_true,2), size(y_rec,2)]);
else
    numcols = size(y_rec,2);
end

for i = 1:numcols
    ax = [ax, subplot(1, numcols+1, i)]; hold on;
    plot(y_rec(:,i), 'LineStyle', '-',  'LineWidth', 1.5);
    if ~isempty(y_true)
        plot(y_true(:,i), 'LineStyle', '--', 'LineWidth', 1.5);
    end
    grid on; hold off; title(['Level ' int2str(i)]);
end
ax = [ax, subplot(1, numcols+1, numcols+1)];
plot(sum(y_rec,2), 'LineStyle', '-', 'LineWidth', 1.5); 
grid on, hold on
plot(y, 'LineStyle', '--', 'LineWidth', 1.5); 
legend('rec.', 'orig.');
title('Reconstructed Signal');

linkaxes(ax, 'xy');    
xlim([find(sum(y_rec,2),1,'first'), find(sum(y_rec,2),1,'last')])


try input('Press Enter...\n\n'); catch, end
try close(fig); catch, end

end

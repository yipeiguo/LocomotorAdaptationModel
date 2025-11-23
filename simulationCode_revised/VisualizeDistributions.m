% This function generates a figure showing how distribution of certain
% features of the trajectory (for different parameter combinations)

function VisualizeDistributions(qMat_cell, qNames_cell, ...
    iflog10vec, ifseparatefig, numbins)

    if ~exist('numbins','var')
        numbins = 10;
    end
    % number of quantities of interest
    numq = length(qMat_cell);
    if ifseparatefig == false
        figure;
        numrows = ceil(sqrt(numq));
        numcols = ceil(numq/numrows);
    end
    
    for qIndx = 1:numq
        if ifseparatefig == false
            subplot(numrows,numcols,qIndx);
        else
            figure;
        end
        
        qMat = qMat_cell{qIndx};
        if iflog10vec(qIndx) == true
            valueMat = log10(qMat);
        else
            valueMat = qMat;
        end
        numcombs = size(qMat,1);
        for combIndx = 1:numcombs
            histogram(valueMat(combIndx,:),numbins,'Normalization','pdf','facealpha',0.5);
            hold on
            [f, xi] = ksdensity(valueMat(combIndx,:));
            plot(xi, f, 'LineWidth', 1);
            hold on
        end
        if iflog10vec(qIndx) == true
            xlabel(strcat('log10(',qNames_cell{qIndx},')'));
        else
            xlabel(qNames_cell{qIndx})
        end
        ylabel('P');
        
    end
    
end

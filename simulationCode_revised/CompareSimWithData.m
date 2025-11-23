% This is a function for comparing trip property distributions between data
% and simulations.


function [KLdiv, overlapFrac] = CompareSimWithData(simdatavec,exptdatavec,numbins)

    % first bin experimental data 
    [countvec_expt,binEdges] = histcounts(exptdatavec,numbins);
    binWidth = binEdges(2) - binEdges(1);
    
    % Get distribution from simulation data using the same binning
    simdatamin = min(simdatavec);
    simdatamax = max(simdatavec);
    if simdatamin < binEdges(1)
        numexcessbins_lower = ceil((binEdges(1)-simdatamin)/binWidth);
    else
        numexcessbins_lower = 0;
    end
    if simdatamax > binEdges(end)
        numexcessbins_upper = ceil((simdatamax-binEdges(end))/binWidth);
    else
        numexcessbins_upper = 0;
    end
    binEdges_new = [binEdges(1) - (numexcessbins_lower:-1:1).*binWidth,...
        binEdges,binEdges(end) + (1:1:numexcessbins_upper).*binWidth];
    Ncounts = histcounts(simdatavec,binEdges_new);
    if min(Ncounts) == 0 % add dummy counts if necessary
        Ncounts = Ncounts + 1;
    end
    pdfVec_sim = Ncounts./sum(Ncounts)./binWidth;
        
    % get full experimental pdf 
    countvec_expt = [zeros(1,numexcessbins_lower),countvec_expt,zeros(1,numexcessbins_upper)]; 
    pdfVec_expt = countvec_expt./sum(countvec_expt)./binWidth;

    % Get KL divergence of data distribution from the simulated
    % distribution
    KLdivVec = pdfVec_expt.*log(pdfVec_expt./pdfVec_sim);
    KLdivVec(pdfVec_expt==0) = 0;
    KLdiv = sum(KLdivVec);

    % get fraction of overlap area between simulation and experimental distributions
    lbVec = min(pdfVec_sim,pdfVec_expt);
    overlapFrac = sum(lbVec)*binWidth;



end
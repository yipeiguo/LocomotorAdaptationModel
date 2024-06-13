% This is a function that defines the type 1 generalized logistic pdf

function pdfscan = genlogisticPDF(xscan,a,loc,scale)

    xscaledVec = (xscan-loc)./scale;
    pdfscan = (a.*exp(-xscaledVec)./(1+exp(-xscaledVec)).^(a+1))./scale;

end
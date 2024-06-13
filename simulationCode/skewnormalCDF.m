% This is a function that defines the type 1 generalized logistic CDF

function CDFscan = skewnormalCDF(xscan,a,loc,scale)

    xscaledVec = (xscan-loc)./scale;
    CDFscan = 0.5.*(1+erf(xscaledVec./sqrt(2))) - 2.*TfnOwen(xscaledVec,a);

end
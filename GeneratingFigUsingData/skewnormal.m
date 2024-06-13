% This is a function that defines the skewnormal pdf

function pdfscan = skewnormal(xscan,a,loc,scale)

    xscaledVec = (xscan-loc)./scale;
    pdfscan = (2.*normpdf(xscaledVec).*0.5.*(1+erf(a.*xscaledVec./sqrt(2))))./scale;

end
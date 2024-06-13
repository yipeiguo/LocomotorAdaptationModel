% This is a function for drawing from skewnormal distribution

function rnd = skewnormalrnd(a,loc,scale,nsamples)

    u = rand(nsamples,1);
    delta = a./sqrt(1+a.^2);
    mu = loc + scale.*delta.*sqrt(2/pi);
    
    rnd = zeros(nsamples,1);
    options = optimset('Display','off');
    for kk = 1:nsamples
        rnd(kk) = fsolve(@(x) u(kk) - skewnormalCDF(x,a,loc,scale),mu,options);
    end

end
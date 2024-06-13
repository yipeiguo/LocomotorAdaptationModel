% This is a function for drawing from type1 genlogistic distribution

function rnd = genlogisticrnd(a,loc,scale,nsamples)

    u = rand(nsamples,1);
    xtildeVec = -log(u.^(-1/a)-1);
    rnd = scale.*xtildeVec + loc;

end
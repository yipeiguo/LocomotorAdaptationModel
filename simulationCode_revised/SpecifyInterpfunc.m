% This is a function that specifies the interpolation function depending on
% the specified type

function [interpfunc,Nparams] = SpecifyInterpfunc(interpFuncType)

    if strcmp(interpFuncType,'exponential')
        interpfunc = @(t,v0,vf,alpha) vf + (v0-vf).*exp(-alpha.*t);
        Nparams = 1;
    elseif strcmp(interpFuncType,'logistic')
        interpfunc = @(t,v0,vf,alpha) vf./(1 + (vf/v0-1).*exp(-alpha.*t));
        Nparams = 1;
    elseif strcmp(interpFuncType,'hill')
        interpfunc = @(t,v0,vf,paramsVec) v0 + ...
            (vf-v0).*t.^paramsVec(2)./(paramsVec(1).^paramsVec(2)+t.^paramsVec(2));
        Nparams = 2;
    end 

end
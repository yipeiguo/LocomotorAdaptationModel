% This is a function that outputs run and turn segment properties 
% from interpolation between very short trips and longer trips.

% Inputs:
% - condOI: (string) data from which fly condition to be used for distributions
% - datatypes_initfinal: (1x2 cell, with each element being a string)
% indicates the datatype used for initial and final parameter values
% - interpFuncType: (string) interpolation function type. can be either 
% 'exponential' or 'logistic' or 'hill'
% - interpParamScan: (#interpParams x #parameterSets array) where 
% #interpParams is 1 for interpFuncType 'exponential' and 'logistic', while
% #interpParams is 2 for interpFuncType 'hill'.

function [runProps_allsets, turnProps_allsets, runProps_initfinal, ...
    turnProps_initfinal, interpFuncs_cell, interpFunc_pCW_cell] = ...
    GetSegProps_interpolated(condOI, datatypes_initfinal, interpFuncType, interpParamScan)

    % Extract variables
    numParamSets = size(interpParamScan,2);
    runProps_allsets = cell(1,numParamSets);
    turnProps_allsets = cell(1,numParamSets);

    % Get run and turn properties of limiting distributions in data
    runProps_initfinal = cell(1,2);
    turnProps_initfinal = cell(1,2);
    for datatypeIndx = 1:2
        datatype = datatypes_initfinal{datatypeIndx};
        [runProps,turnProps] = GetSegProps_multipleDataTypes(condOI, datatype);
        runProps_initfinal{datatypeIndx} = runProps;
        turnProps_initfinal{datatypeIndx} = turnProps;
    end

    % Extract initial and final parameter values for parameters of statistical distributions
    % the distribution of locomotor adaptation distributions are ordered as
    % follows: log10(RL), |turnangle|, log10(radius of turns), P(preferred turn
    % direction).
    % - log10(RL) are assumed to be drawn from skewnormal distribution with 3
    % parameters (a, loc, scale)
    % - |turnangle| are assumed to be drawn from lognormal distribution with 3
    % parameters (s, loc, scale)
    % - log10(rad) are assumed to be drawn from generalized logistic 
    % distribution with 3 parameters (a, loc, scale)
    distparams_initfinal = zeros(3,3,2); % The last dimension is for min/max values
    % probability of turning in preferred direction
    pCW_initfinal = zeros(1,2);
    for datatypeIndx = 1:2
        
        % for P(log10RL):
        runProps = runProps_initfinal{datatypeIndx};
        distparams_initfinal(1,1,datatypeIndx) = runProps.Plog10RL_aFunc(inf);
        distparams_initfinal(1,2,datatypeIndx) = runProps.Plog10RL_locFunc(inf);
        distparams_initfinal(1,3,datatypeIndx) = runProps.Plog10RL_scaleFunc(inf);
    
        % for turn angle and radius
        turnProps = turnProps_initfinal{datatypeIndx};
        % for P(|turnangle|):
        distparams_initfinal(2,1,datatypeIndx) = turnProps.Pturnangle_sFunc(inf);
        distparams_initfinal(2,2,datatypeIndx) = turnProps.Pturnangle_locFunc(inf);
        distparams_initfinal(2,3,datatypeIndx) = turnProps.Pturnangle_scaleFunc(inf);
    
        % for P(log10(rad)):
        distparams_initfinal(3,1,datatypeIndx) = turnProps.Plog10rad_aFunc(inf);
        distparams_initfinal(3,2,datatypeIndx) = turnProps.Plog10rad_locFunc(inf);
        distparams_initfinal(3,3,datatypeIndx) = turnProps.Plog10rad_scaleFunc(inf);
    
        % for pCW:
        pCW_initfinal(datatypeIndx) = turnProps.pCWfunc(inf);
    
    end

    % interpolation function: function used to interpolate between the initial
    % and final values of the parameters
    [interpfunc,~] = SpecifyInterpfunc(interpFuncType);

    % Specify interpolated functions for all distributions and pCW
    interpFuncs_cell = cell(3,3,numParamSets);
    
    for distIndx = 1:3
        for paramIndx = 1:3
            v0 = distparams_initfinal(distIndx,paramIndx,1);
            vf = distparams_initfinal(distIndx,paramIndx,2);
            for setIndx = 1:numParamSets
                paramsVec = interpParamScan(:,setIndx);
                interpFuncs_cell{distIndx,paramIndx,setIndx} = @(t) interpfunc(t,v0,vf,paramsVec);
            end
        end
    end
    
    interpFunc_pCW_cell = cell(1,numParamSets);
    for setIndx = 1:numParamSets
        paramsVec = interpParamScan(:,setIndx);            
        interpFunc_pCW_cell{setIndx} = @(t) interpfunc(t,pCW_initfinal(1),pCW_initfinal(2),paramsVec);
    end

    % transform variables onto the standard mu and sigma parameters of
    % the lognormal distribution.
    % sigma = s
    % mu = log(scale)
    % we draw xtilde from this lognormal, and x = log(turnangle) = xtilde + loc 
    muFunc = @(scale) log(scale);
    
    

    % construct run and turn properties
    for setIndx = 1:numParamSets
        runProps = struct('Plog10RL_aFunc',interpFuncs_cell{1,1,setIndx},...
            'Plog10RL_locFunc',interpFuncs_cell{1,2,setIndx},...
            'Plog10RL_scaleFunc',interpFuncs_cell{1,3,setIndx});

        Pturnangle_sFunc = interpFuncs_cell{2,1,setIndx};
        Pturnangle_scaleFunc = interpFuncs_cell{2,3,setIndx};
        turnProps = struct('Pturnangle_sFunc',Pturnangle_sFunc, ...
            'Pturnangle_locFunc',interpFuncs_cell{2,2,setIndx}, ...
            'Pturnangle_scaleFunc',Pturnangle_scaleFunc, ...
            'Pturnangle_muFunc',@(t) muFunc(Pturnangle_scaleFunc(t)), ...
            'Pturnangle_sigmaFunc',@(t) Pturnangle_sFunc(t), ...
            'Plog10rad_aFunc', interpFuncs_cell{3,1,setIndx}, ...
            'Plog10rad_locFunc', interpFuncs_cell{3,2,setIndx}, ...
            'Plog10rad_scaleFunc', interpFuncs_cell{3,3,setIndx}, ...
            'pCWfunc', interpFunc_pCW_cell{setIndx});

        runProps_allsets{setIndx} = runProps;
        turnProps_allsets{setIndx} = turnProps;
    end

end


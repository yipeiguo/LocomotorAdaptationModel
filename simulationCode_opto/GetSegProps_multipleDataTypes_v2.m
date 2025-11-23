% This is a function that extracts relevant run and turn segment properties 
% from stored data.
% In this version, we include the option to specify which type of data we
% use to specify the statistics of run and turn segments.

% This v2 is an adaptation of 'GetSegProps_multipleDataTypes.m' in the
% folder 'simulationCode_revised'. 
% In this version, we include data for optogenetic flies.

function [runProps,turnProps] = GetSegProps_multipleDataTypes_v2(condOI, datatype)

    if ~exist('datatype','var')
        datatype = 'allTrips_withDistDependence';
    end

    % Summary of data:
    conditions = {'0-125M_24hr', '0-125M_40hr', 'Gr43a-40hr'};
    numconds = length(conditions);

    if strcmp(datatype,'allTrips_withDistDependence')
    
        % distribution of run displacement as a function of total distance
        % log10(rundisp) fits well to a skewnormal distribution
        % initial parameters (from the first run): [a,loc,scale]
        Plog10RL_initparams = [-1.804,0.418,0.611; 1.083, -0.252,0.524;...
            1.145, -0.588, 0.605];
        
        % how parameters vary with distance since last food spot visit:
        Plog10RL_aFunc_cell = {@(t) min(-3.06 + 964./(1+46300.*t).^0.5,Plog10RL_initparams(1,1)),...
            @(t) min(-2.5 + 555./(1+40900.*t).^0.523,Plog10RL_initparams(2,1)),...
            @(t) -33 + 33.4./(1+724.*t).^0.00585};
            % @(t) min(-33 + 33.4./(1+724.*t).^0.00585,Plog10RL_initparams(3,1))};
        Plog10RL_locFunc_cell = {@(t) max(1.13-7.57./(1+10.5.*t).^0.587,Plog10RL_initparams(1,2)),...
            @(t) max(1.1-1.57./(1+3.15.*t).^0.28,Plog10RL_initparams(2,2)),...
            @(t) 3.06-3.45./(1+2.05.*t).^0.051};
            % @(t) max(3.06-3.45./(1+2.05.*t).^0.051,Plog10RL_initparams(3,2))};
        Plog10RL_scaleFunc_cell = {@(t) max(0.769-62.5./(1+50800.*t).^0.451,Plog10RL_initparams(1,3)),...
            @(t) max(0.737-39.2./(1+37500.*t).^0.458,Plog10RL_initparams(2,3)),...
            @(t) 0.7-0.2.*exp(-0.00224.*t)};
    
        
        % distribution of turn angles and how it changes 
        % Here we fitted a lognormal distribution 
        Pturnangle_sFunc_cell = {@(t) 0.788, @(t) 0.867, ...
            @(t) 0.812 + 0.259.*exp(-0.0451.*t)};
        Pturnangle_locFunc_cell = {@(t) -0.0432-0.0259.*exp(-0.0385.*t), ...
            @(t) -0.0308-0.0396.*exp(-0.139.*t), ...
            @(t) -0.094 + 0.0568.*exp(-0.0304.*t)};
        Pturnangle_scaleFunc_cell = {@(t) 0.421 + 0.228.*exp(-0.09.*t), ...
            @(t) 0.352 + 0.3.*exp(-0.113.*t), ...
            @(t) 0.667 - 0.105.*exp(-0.00294.*t)};

        % distribution of radius of curvature of turn segments
        % log10(radius) fitted to a type1 generalized logistic distribution
        Plog10rad_aFunc_cell = {@(t) 0.916 + 4.13.*exp(-0.136.*t),...
            @(t) 1.87 + 74.9.*exp(-0.912.*t), ...
            @(t) 1.05 + 1.53.*exp(-0.00908.*t)};
        Plog10rad_locFunc_cell = {@(t) 0.738 - 1.05.*exp(-0.069.*t),...
            @(t) 0.378 - 1.34.*exp(-0.174.*t), ...
            @(t) 0.432 - 0.748.*exp(-0.0132.*t)};
        Plog10rad_scaleFunc_cell = {@(t) 0.194 + 0.0555.*exp(-0.0458.*t),...
            @(t) 0.262, @(t) 0.23 + 0.114.*exp(-0.0132.*t)};


        % probability of turning in prefered direction or turn bias 
        % as a function of displacement
        pCWfuncs_cell = {@(t) 0.581 + 0.146.*exp(-0.0269.*t), ...
            @(t) 0.56+ 0.181.*exp(-0.0139.*t), ...
            @(t) 0.528 + 0.0917.*exp(-0.023.*t)};
        
        % pbiasFuncs_cell = {@(t) 2.*pCWfuncs_cell{1}(t)-1, ...
        %     @(t) 2.*pCWfuncs_cell{2}(t)-1};

    elseif strcmp(datatype,'shortTrips_withDistDependence')

        % distribution of run displacement as a function of total distance
        % log10(rundisp) fits well to a skewnormal distribution
        % initial parameters (from the first run): [a,loc,scale]
        Plog10RL_initparams = [-1.01,0.216,0.476; -1.283, 0.26,0.502; ...
            -1.08, 0.014, 0.511];
        
        % how parameters vary with distance since last food spot visit:
        Plog10RL_aFunc_cell = {@(t) min(-2.64 + 1.22.*exp(-0.0364.*t),Plog10RL_initparams(1,1)),...
            @(t) -1.55 + 100.*exp(-0.11.*t) - 99.6.*exp(-0.107.*t),...
            @(t) -1.53 + 1.33.*exp(-0.15.*t)};
        Plog10RL_locFunc_cell = {@(t) 1-0.794.*exp(-0.0413.*t),...
            @(t) 0.605-0.435.*exp(-0.108.*t),...
            @(t) 0.338 - 0.62.*exp(-0.0837.*t)};
        Plog10RL_scaleFunc_cell = {@(t) 0.664-0.212.*exp(-0.208.*t),...
            @(t) 0.33 - 0.237.*exp(-0.191.*t) + 0.366.*exp(-0.00545.*t), ...
            @(t) 0.552 - 0.128.*exp(-0.12.*t)};
        
        % distribution of turn angles and how it changes 
        % Here we fitted a lognormal distribution 
        Pturnangle_sFunc_cell = {@(t) 0.755 + 0.165.*exp(-0.713.*t), ...
            @(t) 0.881, @(t) 0.853 + 0.269.*exp(-0.0868.*t)};
        Pturnangle_locFunc_cell = {@(t) -0.089 + 0.0725.*exp(-1.61.*t), ...
            @(t) -0.0128-0.0601.*exp(-0.0307.*t), ...
            @(t) -0.0918 + 0.0591.*exp(-0.0421.*t)};
        Pturnangle_scaleFunc_cell = {@(t) 0.437 + 0.287.*exp(-0.0214.*t), ...
            @(t) 0.383 + 0.376.*exp(-0.049.*t), ...
            @(t) 0.58};

    
        % distribution of radius of curvature of turn segments
        % log10(radius) fitted to a type1 generalized logistic distribution
        Plog10rad_aFunc_cell = {@(t) 1.18 + 7.96.*exp(-0.229.*t),...
            @(t)  1.96 + 565.*exp(-1.36.*t), ...
            @(t) -0.031 + 2.56.*exp(-0.00496.*t)};
        Plog10rad_locFunc_cell = {@(t) 0.659 - 1.04.*exp(-0.0686.*t),...
            @(t) 0.346 - 1.83.*exp(-0.24.*t), ...
            @(t) 0.482 - 0.833.*exp(-0.017.*t)};
        Plog10rad_scaleFunc_cell = {@(t) 0.184 + 0.0571.*exp(-0.0222.*t),...
            @(t) 0.26, @(t) 0.237 + 0.122.*exp(-0.0298.*t)};
        % Plog10rad_scaleFunc_cell = {@(t) 0.184 + 0.0571.*exp(-0.0222.*t),...
        %     @(t) 0.251 + 0.0135.*exp(-0.261.*t)};


        % probability of turning in prefered direction or turn bias 
        % as a function of displacement
        pCWfuncs_cell = {@(t) 0.623 - 0.335.*exp(-0.353.*t) + 0.331.*exp(-0.0415.*t), ...
            @(t) 0.5 - 0.265.*exp(-0.61.*t) + 0.38.*exp(-0.01.*t), ...
            @(t) 0.556 + 0.0973.*exp(-0.0305.*t)};
        % for 24hr condition, if we use only flies with Preturn>0.7,with 18
        % bins: pCW(t) = 0.583 - 0.266.*exp(-0.45.*t) + 0.318.*exp(-0.0208.*t)

    elseif strcmp(datatype,'veryShortTrips') % no distance dependence
        
        % Here we extract distributions from either:
        % (i) trips with <=5 segments
        % (ii) 40% of the trips with the fewest number of segments 
        % (in the case of WT flies, this corresponds to having <=5 movement 
        % segments. For Gr43a flies, this means having <=19 segments)
        
        % Parameters for run length distributions
        % (i) 
        Plog10RL_aFunc_cell = {@(t) -1.1, @(t) -0.86, @(t) 0.974};
        Plog10RL_locFunc_cell = {@(t) 0.115, @(t) 0.073, @(t) -0.592};
        Plog10RL_scaleFunc_cell = {@(t) 0.52, @(t) 0.438, @(t) 0.507};
        % (ii) 
        % Plog10RL_aFunc_cell = {@(t) -1.1, @(t) -0.86, @(t) -1.895};
        % Plog10RL_locFunc_cell = {@(t) 0.115, @(t) 0.073, @(t) 0.19};
        % Plog10RL_scaleFunc_cell = {@(t) 0.52, @(t) 0.438, @(t) 0.598};
        % (iii) try using RL distributions from all trips for the opto genotype 
        % Plog10RL_aFunc_cell = {@(t) -1.1, @(t) -0.86,  ...
        %     @(t) -33 + 33.4./(1+724.*t).^0.00585};
        % Plog10RL_locFunc_cell = {@(t) 0.115, @(t) 0.073, ...
        %     @(t) 3.06-3.45./(1+2.05.*t).^0.051};
        % Plog10RL_scaleFunc_cell = {@(t) 0.52, @(t) 0.438, ...
        %     @(t) 0.7-0.2.*exp(-0.00224.*t)};


        % distribution of turn angles
        % (i) 
        Pturnangle_sFunc_cell = {@(t) 0.72, @(t) 0.54, @(t) 0.122};
        Pturnangle_locFunc_cell = {@(t) 0.188, @(t) -0.5, @(t) -8.4};
        Pturnangle_scaleFunc_cell = {@(t) 1.154, @(t) 1.66, @(t) 10.21};
        % (ii) 
        % Pturnangle_sFunc_cell = {@(t) 0.72, @(t) 0.54, @(t) 1.045};
        % Pturnangle_locFunc_cell = {@(t) 0.188, @(t) -0.5 @(t) -0.054};
        % Pturnangle_scaleFunc_cell = {@(t) 1.154, @(t) 1.66, @(t) 0.722};
        
        % distribution of radius of curvature of turn segments
        % (i) 
        Plog10rad_aFunc_cell = {@(t) 2.0, @(t) 2.24, @(t) 1.69};
        Plog10rad_locFunc_cell = {@(t) -0.05, @(t) -0.075, @(t) -0.15};
        Plog10rad_scaleFunc_cell = {@(t) 0.19, @(t) 0.215, @(t) 0.37};
        % (ii) 
        % Plog10rad_aFunc_cell = {@(t) 2.0, @(t) 2.24, @(t) 2.15};
        % Plog10rad_locFunc_cell = {@(t) -0.05, @(t) -0.075, @(t) -0.218};
        % Plog10rad_scaleFunc_cell = {@(t) 0.19, @(t) 0.215, @(t) 0.329};

        % Probability of turning in preferred direction
        % (i) 
        pCWfuncs_cell = {@(t) 0.893, @(t) 0.835, @(t) 0.8476};
        % (ii) 
        % pCWfuncs_cell = {@(t) 0.893, @(t) 0.835, @(t) 0.685};

        % From trips with <=3 movement segments and 24 hr condition:
        % runProps.Plog10RL_aFunc = @(t) -1.9;
        % runProps.Plog10RL_locFunc = @(t) 0.25;
        % runProps.Plog10RL_scaleFunc = @(t) 0.6;
        
        % Pturnangle_sFunc = @(t) 0.44;
        % Pturnangle_locFunc = @(t) -0.719;
        % Pturnangle_scaleFunc = @(t) 2.1;

    end


    % transform variables onto the standard mu and sigma parameters of
    % the lognormal distribution.
    % sigma = s
    % mu = log(scale)
    % we draw xtilde from this lognormal, and x = log(turnangle) = xtilde + loc 
    muFunc = @(scale) log(scale);
    
    Pturnangle_muFunc_cell = cell(1,numconds);
    Pturnangle_sigmaFunc_cell = cell(1,numconds);
    for condIndx = 1:numconds
        sFunc = Pturnangle_sFunc_cell{condIndx};
        scaleFunc = Pturnangle_scaleFunc_cell{condIndx};
        Pturnangle_muFunc_cell{condIndx} = @(t) muFunc(scaleFunc(t));
        Pturnangle_sigmaFunc_cell{condIndx} = @(t) sFunc(t);
    end

    

    % Extract relevant data
    condIndxOI = find(strcmp(conditions,condOI),1);
    if ~isempty(condIndxOI)
        % distribution of run lengths for straight-line run segments
        runProps.Plog10RL_aFunc = Plog10RL_aFunc_cell{condIndxOI};
        runProps.Plog10RL_locFunc = Plog10RL_locFunc_cell{condIndxOI};
        runProps.Plog10RL_scaleFunc = Plog10RL_scaleFunc_cell{condIndxOI};
        % turn angles are assumed to be drawn from truncated lognormal
        % distribution with parameters mu and sigma (and shifted by loc).
        turnProps.Pturnangle_sFunc = Pturnangle_sFunc_cell{condIndxOI};
        turnProps.Pturnangle_locFunc = Pturnangle_locFunc_cell{condIndxOI};
        turnProps.Pturnangle_scaleFunc = Pturnangle_scaleFunc_cell{condIndxOI};
        turnProps.Pturnangle_muFunc = Pturnangle_muFunc_cell{condIndxOI};
        turnProps.Pturnangle_sigmaFunc = Pturnangle_sigmaFunc_cell{condIndxOI};
        % distribution of radius of curvature for turns:
        turnProps.Plog10rad_aFunc = Plog10rad_aFunc_cell{condIndxOI};
        turnProps.Plog10rad_locFunc = Plog10rad_locFunc_cell{condIndxOI};
        turnProps.Plog10rad_scaleFunc = Plog10rad_scaleFunc_cell{condIndxOI};
        % how probability of turning in preferred direction changes over trip:
        turnProps.pCWfunc = pCWfuncs_cell{condIndxOI};
    
    else
        warning('desired condition not found!');
        runProps = [];
        turnProps = [];
    end
    

end
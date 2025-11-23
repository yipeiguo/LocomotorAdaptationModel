% This is a function that extracts relevant run and turn segment properties 
% from stored data.
% In this version, we include the option to specify which type of data we
% use to specify the statistics of run and turn segments.

function [runProps,turnProps] = GetSegProps_multipleDataTypes(condOI, datatype)

    if ~exist('datatype','var')
        datatype = 'allTrips_withDistDependence';
    end

    % Summary of data:
    conditions = {'0-125M_24hr', '0-125M_40hr'};
    numconds = length(conditions);

    if strcmp(datatype,'allTrips_withDistDependence')
    
        % distribution of run displacement as a function of total distance
        % log10(rundisp) fits well to a skewnormal distribution
        % initial parameters (from the first run): [a,loc,scale]
        Plog10RL_initparams = [-1.804,0.418,0.611; 1.083, -0.252,0.524];
        
        % how parameters vary with distance since last food spot visit:
        Plog10RL_aFunc_cell = {@(t) min(-3.06 + 964./(1+46300.*t).^0.5,Plog10RL_initparams(1,1)),...
            @(t) min(-2.5 + 555./(1+40900.*t).^0.523,Plog10RL_initparams(2,1))};
        Plog10RL_locFunc_cell = {@(t) max(1.13-7.57./(1+10.5.*t).^0.587,Plog10RL_initparams(1,2)),...
            @(t) max(1.1-1.57./(1+3.15.*t).^0.28,Plog10RL_initparams(2,2))};
        Plog10RL_scaleFunc_cell = {@(t) max(0.769-62.5./(1+50800.*t).^0.451,Plog10RL_initparams(1,3)),...
            @(t) max(0.737-39.2./(1+37500.*t).^0.458,Plog10RL_initparams(2,3))};
    
        
        % distribution of turn angles and how it changes 
        % Here we fitted a lognormal distribution 
        Pturnangle_sFunc_cell = {@(t) 0.788, @(t) 0.867};
        Pturnangle_locFunc_cell = {@(t) -0.0432-0.0259.*exp(-0.0385.*t), ...
            @(t) -0.0308-0.0396.*exp(-0.139.*t)};
        Pturnangle_scaleFunc_cell = {@(t) 0.421 + 0.228.*exp(-0.09.*t), ...
            @(t) 0.352 + 0.3.*exp(-0.113.*t)};

        % distribution of radius of curvature of turn segments
        % log10(radius) fitted to a type1 generalized logistic distribution
        Plog10rad_aFunc_cell = {@(t) 0.916 + 4.13.*exp(-0.136.*t),...
            @(t) 1.87 + 74.9.*exp(-0.912.*t)};
        Plog10rad_locFunc_cell = {@(t) 0.738 - 1.05.*exp(-0.069.*t),...
            @(t) 0.378 - 1.34.*exp(-0.174.*t)};
        Plog10rad_scaleFunc_cell = {@(t) 0.194 + 0.0555.*exp(-0.0458.*t),...
            @(t) 0.262};


        % probability of turning in prefered direction or turn bias 
        % as a function of displacement
        pCWfuncs_cell = {@(t) 0.581 + 0.146.*exp(-0.0269.*t), ...
            @(t) 0.56+ 0.181.*exp(-0.0139.*t)};
        
        % pbiasFuncs_cell = {@(t) 2.*pCWfuncs_cell{1}(t)-1, ...
        %     @(t) 2.*pCWfuncs_cell{2}(t)-1};

    elseif strcmp(datatype,'shortTrips_withDistDependence')

        % distribution of run displacement as a function of total distance
        % log10(rundisp) fits well to a skewnormal distribution
        % initial parameters (from the first run): [a,loc,scale]
        Plog10RL_initparams = [-1.01,0.216,0.476; -1.283, 0.26,0.502];
        
        % how parameters vary with distance since last food spot visit:
        % Plog10RL_aFunc_cell = {@(t) min(-3.25 + 2.58./(1+0.099.*t).^0.57,Plog10RL_initparams(1,1)),...
        %     @(t) min(-1.55 + 100.*exp(-0.11.*t) - 99.6.*exp(-0.107.*t),Plog10RL_initparams(2,1))};
        % Plog10RL_locFunc_cell = {@(t) max(1.1-1.08./(1+0.0183.*t).^2.97,Plog10RL_initparams(1,2)),...
        %     @(t) max(0.605-0.435.*exp(-0.108.*t),Plog10RL_initparams(2,2))};
        % Plog10RL_scaleFunc_cell = {@(t) max(0.719-0.32./(1+0.171.*t).^1.1,Plog10RL_initparams(1,3)),...
        %     @(t) 0.33 - 0.237.*exp(-0.191.*t) + 0.366.*exp(-0.00545.*t)};
        % For 24 hr flies, if we only look at flies with Preturn>0.7:
        Plog10RL_aFunc_cell = {@(t) min(-2.64 + 1.22.*exp(-0.0364.*t),Plog10RL_initparams(1,1)),...
            @(t) -1.55 + 100.*exp(-0.11.*t) - 99.6.*exp(-0.107.*t)};
        Plog10RL_locFunc_cell = {@(t) 1-0.794.*exp(-0.0413.*t),...
            @(t) 0.605-0.435.*exp(-0.108.*t)};
        Plog10RL_scaleFunc_cell = {@(t) 0.664-0.212.*exp(-0.208.*t),...
            @(t) 0.33 - 0.237.*exp(-0.191.*t) + 0.366.*exp(-0.00545.*t)};
        % Plog10RL_scaleFunc_cell = {@(t) max(0.719-0.32./(1+0.171.*t).^1.1,Plog10RL_initparams(1,3)),...
        %     @(t) max(0.33 - 0.237.*exp(-0.191.*t) + 0.366.*exp(-0.00545.*t),Plog10RL_initparams(2,3))};

        % distribution of turn angles and how it changes 
        % Here we fitted a lognormal distribution 
        Pturnangle_sFunc_cell = {@(t) 0.755 + 0.165.*exp(-0.713.*t), ...
            @(t) 0.881};
        Pturnangle_locFunc_cell = {@(t) -0.089 + 0.0725.*exp(-1.61.*t), ...
            @(t) -0.0128-0.0601.*exp(-0.0307.*t)};
        Pturnangle_scaleFunc_cell = {@(t) 0.437 + 0.287.*exp(-0.0214.*t), ...
            @(t) 0.383 + 0.376.*exp(-0.049.*t)};

    
        % distribution of radius of curvature of turn segments
        % log10(radius) fitted to a type1 generalized logistic distribution
        Plog10rad_aFunc_cell = {@(t) 1.18 + 7.96.*exp(-0.229.*t),...
            @(t)  1.96 + 565.*exp(-1.36.*t)};
        Plog10rad_locFunc_cell = {@(t) 0.659 - 1.04.*exp(-0.0686.*t),...
            @(t) 0.346 - 1.83.*exp(-0.24.*t)};
        Plog10rad_scaleFunc_cell = {@(t) 0.184 + 0.0571.*exp(-0.0222.*t),...
            @(t) 0.26};
        % Plog10rad_scaleFunc_cell = {@(t) 0.184 + 0.0571.*exp(-0.0222.*t),...
        %     @(t) 0.251 + 0.0135.*exp(-0.261.*t)};


        % probability of turning in prefered direction or turn bias 
        % as a function of displacement
        pCWfuncs_cell = {@(t) 0.623 - 0.335.*exp(-0.353.*t) + 0.331.*exp(-0.0415.*t), ...
            @(t) 0.5 - 0.265.*exp(-0.61.*t) + 0.38.*exp(-0.01.*t)};
        % for 24hr condition, if we use only flies with Preturn>0.7,with 18
        % bins: pCW(t) = 0.583 - 0.266.*exp(-0.45.*t) + 0.318.*exp(-0.0208.*t)

    elseif strcmp(datatype,'shortTrips_withDistDependence_v2')
        % long/short trips classified using distribution of scaled distance
        % to food or border

        % run lengths:
        Plog10RL_aFunc_cell = {@(t) -2.14 + 3.23.*exp(-0.308.*t),...
            @(t) -2};
        Plog10RL_locFunc_cell = {@(t) 0.855 - 1.21.*exp(-0.139.*t),...
            @(t) 0.639-0.442.*exp(-0.0816.*t)};
        Plog10RL_scaleFunc_cell = {@(t) 0.721-0.347.*exp(-0.195.*t),...
            @(t) 0.681-0.208.*exp(-0.242.*t)};

        % turn angles 
        Pturnangle_sFunc_cell = {@(t) 0.915.*exp(-0.177.*t) - 0.83.*(exp(-0.102.*t)-1), ...
            @(t) 0.985.*exp(-0.807.*t) - 0.856.*(exp(-0.506.*t)-1)};
        Pturnangle_locFunc_cell = {@(t) 0.018.*t + 0.159.*exp(-0.174.*t), ...
            @(t) -0.0558 + 27.1.*(exp(-0.502.*t)-exp(-0.499.*t))};
        Pturnangle_scaleFunc_cell = {@(t) 0.488 - 0.427.*exp(-0.916.*t) + 0.454.*exp(-0.0404.*t), ...
            @(t) 0.63.*exp(-0.129.*t) - 0.475.*(exp(-0.675.*t)-1)};

        % log10(radius)
        Plog10rad_aFunc_cell = {@(t) 1.18 + 7.96.*exp(-0.229.*t),...
            @(t)  1.96 + 565.*exp(-1.36.*t)};
        Plog10rad_locFunc_cell = {@(t) 0.659 - 1.04.*exp(-0.0686.*t),...
            @(t) 0.346 - 1.83.*exp(-0.24.*t)};
        Plog10rad_scaleFunc_cell = {@(t) 0.184 + 0.0571.*exp(-0.0222.*t),...
            @(t) 0.26};
        
        
        % pCW
        pCWfuncs_cell = {@(t) 0.671.*exp(-0.0584.*t) - 0.616.*(exp(-0.208.*t)-1), ...
            @(t) 0.607.*exp(-0.133.*t) - 0.79.*(exp(-0.296.*t)-1)};
        

    elseif strcmp(datatype,'veryShortTrips') % no distance dependence
        
        % Here we extract distributions from trips with <= 5 movement segments
        % Parameters for run length distributions
        Plog10RL_aFunc_cell = {@(t) -1.1, @(t) -0.86};
        Plog10RL_locFunc_cell = {@(t) 0.115, @(t) 0.073};
        Plog10RL_scaleFunc_cell = {@(t) 0.52, @(t) 0.438};

        % distribution of turn angles
        Pturnangle_sFunc_cell = {@(t) 0.72, @(t) 0.54};
        Pturnangle_locFunc_cell = {@(t) 0.188, @(t) -0.5};
        Pturnangle_scaleFunc_cell = {@(t) 1.154, @(t) 1.66};

        % distribution of radius of curvature of turn segments
        Plog10rad_aFunc_cell = {@(t) 2.0, @(t) 2.24};
        Plog10rad_locFunc_cell = {@(t) -0.05, @(t) -0.075};
        Plog10rad_scaleFunc_cell = {@(t) 0.19, @(t) 0.215};

        % Probability of turning in preferred direction
        pCWfuncs_cell = {@(t) 0.893, @(t) 0.835};

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
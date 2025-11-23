% In this a script for simulating trajectories using run and turn
% segment distributions extracted from the data.

% The data-constrained input distributions of locomotor parameters here are
% obtained from and interpolation between that of very short trips and that
% of longer trips.

% Here the log10(run length) are drawn from a skewnormal distribution with
% parameters that vary over a trip, the turn angle distributions are drawn
% from a truncated lognormal distribution with parameters that vary over a
% trip, log10(radius of curvature of turn segments) are drawn from a
% type 1 generalized logistic distribution, and the probability of turning 
% clockwise (assumed to be the preferred direction) varies during a trip.

close all;
clear variables;
rng('shuffle');


% How mean and variance of log10(rad) depends on these parameters
genlogistic_meanFunc = @(a,loc,scale) scale.*(psi(a)-psi(1)) + loc;
genlogistic_varFunc = @(a,loc,scale) (scale.^2).*(psi(1,a)-psi(1,1));

% For lognormal distribution
muFunc_lognorm = @(scale) log(scale);

% Import experimental data for comparison with simulations
% Experimental data for distributions of trip properties
tripPropNames = {'numSegs_toFoodOrBorder','distance_toFoodOrBorder','max_disp'};
numTripProps = length(tripPropNames);
tripPropDataCell = cell(1,numTripProps);
ifdatalogged = false;
for propIndx = 1:numTripProps
    tripProp = tripPropNames{propIndx};
    if ifdatalogged == true
        fn = strcat('log10(',tripProp,')_FlyData_0-125M_24hr');
    else
        fn = strcat(tripProp,'_FlyData_0-125M_24hr');
    end
    data_imported = load(strcat('../figures/',fn,'.csv'));
    if ifdatalogged == true
        tripPropDataCell{propIndx} = data_imported;
    else
        tripPropDataCell{propIndx} = log10(data_imported);
    end
end



%% Define parameters
% radius of food patch (which is assumed to be at origin):
eps = 1.5; 
initposParams.eps = eps;
% radius of arena, which is assumed to be circular
radius_arena = 80;
initposParams.radius_arena = radius_arena;
% Here we allow for additional detection zone such that the agent can be
% considered to have returned even if it doesn't cross into the food spot.
% In the context of the fly, this could be ~half the body length (to
% account for the difference in body vs head position
sensingdist = 1;
ifdiffCWvsAntiCW = false; % whether we have different statistics for CW vs antiCW turns

% technical simulation parameters:
maxdist = 4e4; % maximum distance travelled before simulation stops
maxNumSteps = 4e4;
numtrials = 1e4;


% fn_generalinfo = strcat('_eps_',num2str(eps),'_R_',num2str(radius_arena),...
%     '_maxdist',num2str(maxdist),...
%     '_maxNumSteps',num2str(maxNumSteps),'_numtrials',num2str(numtrials));

% Get run and turn segment properties
condOI = '0-125M_24hr'; % '0-125M_24hr' or '0-125M_40hr'

% datatypes for the initial and final stages of a trajectory:
% possible data types:
% 'allTrips_withDistDependence' or 'shortTrips_withDistDependence' or
% 'veryShortTrips'
datatypes_initfinal = {'veryShortTrips','shortTrips_withDistDependence'};

% interpolation function: function used to interpolate between the initial
% and final values of the parameters
interpFuncType = 'exponential'; % 'exponential' or 'logistic' or 'hill'
interpParamScan = 0.015;
numParamSets = length(interpParamScan);

% Extract run and turn properties
[runProps_allsets, turnProps_allsets, runProps_initfinal, ...
    turnProps_initfinal, interpFuncs_cell, interpFunc_pCW_cell] = ...
    GetSegProps_interpolated(condOI, datatypes_initfinal, interpFuncType, interpParamScan);


% initial probability of run (vs turn)
initPrun = 0.5;

% parameters for initial position and direction
initdirParams.initdirType = 'random'; % 'random' or 'straightOut' or 'specified'
if strcmp(initdirParams.initdirType,'specified') % specify a desired value
    initdirParams.initTurnAngle = -0.25*pi; % -pi/4, which is 45 degrees
end


% specify filename to save
ifsave = false; % whether to save data
if ifsave == true
    fn2save = strcat('SimData','_',condOI,'_interpType_',interpFuncType,'_numtrials',num2str(numtrials));
    foldername = fn2save;
    mkdir(foldername)
    for setIndx = 1:numParamSets
        interpParamOI = interpParamScan(:,setIndx);
        subfoldername = strcat('alpha',num2str(interpParamOI(1)));
        subfoldername = strrep(subfoldername,'.','pt');
        mkdir(strcat(foldername,'/',subfoldername));
    end
    % specify variables to save
    qNames2save = {'totdist','maxdisp','ifreturn','ifhitwall','numsteps'};
    numq2save = length(qNames2save);

    % for Preturn data
    numbins = 100;
    qNames_cell = {'numsteps','maxdisp','totdist','maxdisp'};
    numq = length(qNames_cell);
    % iflogVec = ones(1,numq);
    % iflogVec = zeros(1,numq);
    iflogVec = [1,0,1,1];
    ifexceedVec = [true,true,true,true];
        
end

% for binning data to compare between sim and expt
numbinsVec_data = [20,20,20]; 

%% storage arrays
% fraction of trials that either returned to food or hit arena boundary
relTrialFrac_allsets = zeros(1,numParamSets);
overallPreturn_allsets = zeros(1,numParamSets);
maxDisp_condmean_allsets = zeros(1,numParamSets);
totdist_condmean_allsets = zeros(1,numParamSets);

% comparison between simulation and data distributions
% the last dimension represents which quantification:
% - 1st element: KL divergence
% - 2nd element: overlap fraction between the 2 distributions
distComparisonMat = zeros(numParamSets,numTripProps,2);

%% For each set (of interpolation choice), carry out main simulation,
% save desired simulation results
% due to memory constraints, we will overwrite variables and not store raw
% data
tic
for setIndx = 1:numParamSets
    fprintf('setIndx: %d \n',setIndx);
    runProps = runProps_allsets{setIndx};
    turnProps = turnProps_allsets{setIndx};
    [posTrajs, headangleTrajs, segLengthTrajs, segArcProps_cell, ...
        initsegTypeVec, dispTrajs, distTrajs, totdistVec, maxdispVec, ...
        ifreturnVec, ifhitwallVec, numstepsVec] = ...
        SimTrajs_withCurvedSegs_v2(initposParams, initdirParams, initPrun, runProps, turnProps, eps, ...
        radius_arena, sensingdist, maxdist, maxNumSteps, numtrials);
    
    % Basic analysis and summary statistics
    % fraction of trials that ended because walker encountered either food spot
    % or arena boundary
    numtrials_returned = sum(ifreturnVec);
    numtrials_OI = numtrials_returned + sum(ifhitwallVec);
    trialFrac_OI = numtrials_OI./numtrials;
    relTrialFrac_allsets(setIndx) = trialFrac_OI;
    disp(strcat('relevant trial fraction:',num2str(trialFrac_OI)));
    
    % probability of encountering food spot before wall
    Preturn_overall = numtrials_returned./numtrials_OI;
    overallPreturn_allsets(setIndx) = Preturn_overall;
    disp(strcat('average Preturn:',num2str(Preturn_overall)));
    
    % average maximum displacement given that walker returns to food spot
    maxDisp_condmean = sum(maxdispVec.*ifreturnVec)./numtrials_returned;
    maxDisp_condmean_allsets(setIndx) = maxDisp_condmean;
    
    % average total distance of trip given that walker returns to food spot
    totdist_condmean = sum(totdistVec.*ifreturnVec)./numtrials_returned;
    totdist_condmean_allsets(setIndx) = totdist_condmean;

    % compare distribution of trip properties with that of expts
    tripProp_cell = {log10(numstepsVec),log10(totdistVec),log10(maxdispVec)};
    for tripPropIndx = 1:numTripProps
        [KLdiv, overlapFrac] = CompareSimWithData(tripProp_cell{tripPropIndx},...
            reshape(tripPropDataCell{tripPropIndx},1,[]),numbinsVec_data(tripPropIndx));
        distComparisonMat(setIndx,tripPropIndx,:) = [KLdiv, overlapFrac];
    end

    % save simulation data if desired
    if ifsave == true
        q2save = {totdistVec, maxdispVec, ifreturnVec, ifhitwallVec, numstepsVec};
        interpParamOI = interpParamScan(:,setIndx);
        subfoldername = strcat('alpha',num2str(interpParamOI(1)));
        subfoldername = strrep(subfoldername,'.','pt');
        for qIndx = 1:numq2save
            writematrix(q2save{qIndx},strcat(foldername,'/',subfoldername,'/',qNames2save{qIndx},'.csv'));
        end
    
        % calculate Preturn data 
        quantitiesOI = {numstepsVec, maxdispVec, totdistVec, maxdispVec};
        log10quantitiesOI = cell(1,numq);
        for qIndx = 1:numq
            quantityMat = quantitiesOI{qIndx};
            log10quantitiesOI{qIndx} = log10(quantityMat);
        end
        
        for qIndx = 1:numq
            if iflogVec(qIndx) == 1
                dataVec = log10quantitiesOI{qIndx};
                xname = strcat('log10(',qNames_cell{qIndx},')');
            else
                dataVec = quantitiesOI{qIndx};
                xname = qNames_cell{qIndx};
            end
            [NtrialsVec,BinEdges] = histcounts(dataVec,numbins);
            binmid = (BinEdges(1:end-1)+BinEdges(2:end))./2;
        
            % number of successfully-returned trials 
            if ifexceedVec(qIndx) == false
                NsuccessVec = histcounts(dataVec(ifreturnVec==1),BinEdges);
            else
                NtrialsVec = zeros(1,numbins);
                NsuccessVec = zeros(1,numbins);
                for binIndx = 1:numbins
                    NtrialsVec(binIndx) = sum(dataVec >= binmid(binIndx));
                    NsuccessVec(binIndx) = sum(dataVec(ifreturnVec==1) >= binmid(binIndx));
                end
            end
            PreturnVec = NsuccessVec./NtrialsVec;
            
            meanPreturnMat = [binmid;PreturnVec];
            writematrix(meanPreturnMat,strcat(foldername,'/',subfoldername,'/PreturnVS',xname,'_numbins',num2str(numbins),'.csv'));
        end

    end
    toc
end
toc

%% For different maximum displacement ranges, visualize trajectory with the 
% median total distance 
% Note that since the raw simulation variables are being overwritten for
% every set, this section only plots trajectories for the last set.
log10maxdisp_OI = [0.4,0.8,1.2,1.6,1.8,1.85];
maxdisp_eps = 0.05;
maxdispOI_min = 10.^(log10maxdisp_OI-maxdisp_eps);
maxdispOI_max = 10.^(log10maxdisp_OI+maxdisp_eps);

numMaxDispOI = length(log10maxdisp_OI);
trialsRel = zeros(1,numMaxDispOI);
numPosTrials = zeros(1,numMaxDispOI);
subplotTitles = cell(1,numMaxDispOI);
% select trials:
for kk = 1:numMaxDispOI
    maxdisp_min = maxdispOI_min(kk);
    maxdisp_max = maxdispOI_max(kk);
    postrials = find((maxdispVec>maxdisp_min) & (maxdispVec<maxdisp_max));
    if ~isempty(postrials)
        totdist_pos = totdistVec(postrials);
        [~, idx] = min(abs(totdist_pos-median(totdist_pos)));
        trialsRel(kk) = postrials(idx);
        numPosTrials(kk) = length(postrials);
    else
        [~, idx] = min(abs(totdistVec-10.^log10maxdisp_OI(kk)));
        numPosTrials(kk) = 1;
        trialsRel(kk) = idx;
    end
    subplotTitles{kk} = strcat('log10(maxdisp) = ',num2str(log10maxdisp_OI(kk)));
end

% plot
posTrajsRel = posTrajs(:,:,trialsRel);
numstepsRel = numstepsVec(trialsRel);
ifhitwallRel = ifhitwallVec(trialsRel);
initsegTypeRel = initsegTypeVec(trialsRel);
segArcPropsRel = segArcProps_cell(trialsRel,:);

titlename = strcat('\alpha=',num2str(interpParamScan(end)));
ifplotarena = false;
VisualizeTrajectories_withCurvedSegs(posTrajsRel, segArcPropsRel,...
    initsegTypeRel,eps,numstepsRel,radius_arena,ifhitwallRel,titlename,...
    ifplotarena, subplotTitles);


%% Save storage arrays in corresponding folder for interpolation parameter
if ifsave == true
    metric = ["KLdiv";"overlapFrac"];
    for setIndx = 1:numParamSets
        % construct table for overall simulation properties
        relTrialFrac = relTrialFrac_allsets(setIndx);
        overallPreturn = overallPreturn_allsets(setIndx);
        maxDisp_condmean = maxDisp_condmean_allsets(setIndx);
        totdist_condmean = totdist_condmean_allsets(setIndx);
        table_currset = table(relTrialFrac,overallPreturn,maxDisp_condmean,totdist_condmean);
    
        % save table
        interpParamOI = interpParamScan(:,setIndx);
        subfoldername = strcat('alpha',num2str(interpParamOI(1)));
        subfoldername = strrep(subfoldername,'.','pt');
        writetable(table_currset,strcat(foldername,'/',subfoldername,'/OverallSimProps.csv'));
    
        % construct table for comparison between simulation and data
        numSegsComparison = reshape(distComparisonMat(setIndx,1,:),2,1);
        totdistComparison = reshape(distComparisonMat(setIndx,2,:),2,1);
        maxdispComparison = reshape(distComparisonMat(setIndx,3,:),2,1);
        tablecomparison = table(metric,numSegsComparison,totdistComparison,maxdispComparison);
    
        % save table
        writetable(tablecomparison,strcat(foldername,'/',subfoldername,'/SimExptComparison.csv'));
    
    end
end

%% Plotting

% sanity check for basic properties of simulations
arrays2plot = {relTrialFrac_allsets, overallPreturn_allsets, ...
    maxDisp_condmean_allsets, totdist_condmean_allsets};
arrayNames = {'relevant trial fraction','overallPreturn',...
    'maxDisp_condmean','totdist_condmean'};
figure;
for kk = 1:4
    subplot(2,2,kk);
    plot(interpParamScan,arrays2plot{kk},'x-');
    xlabel('\alpha');
    ylabel(arrayNames{kk});
end

% visualizing the effect of varying the interpolation parameter on
% how well simulation agrees with data
metricNames = {'KLdiv','overlapFrac'};
figure;
for tripPropIndx = 1:numTripProps
    for metricIndx = 1:2
        subplot(2,numTripProps,(metricIndx-1)*numTripProps+tripPropIndx);
        plot(interpParamScan,distComparisonMat(:,tripPropIndx,metricIndx),'x-');
        xlabel('\alpha');
        ylabel(metricNames{metricIndx});
        if metricIndx == 1
            title(tripPropNames{tripPropIndx});
        end
    end    
end














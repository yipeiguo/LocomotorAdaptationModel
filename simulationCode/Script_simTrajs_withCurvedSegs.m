% In this a script for simulating trajectories using run and turn
% segment distributions extracted from the data.

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

% technical simulation parameters:
maxdist = 4e4; % maximum distance travelled before simulation stops
maxNumSteps = 4e4;
numtrials = 1e4;


% fn_generalinfo = strcat('_eps_',num2str(eps),'_R_',num2str(radius_arena),...
%     '_maxdist',num2str(maxdist),...
%     '_maxNumSteps',num2str(maxNumSteps),'_numtrials',num2str(numtrials));

% Get run and turn segment properties
condOI = '0-125M_24hr'; % '0-125M_24hr' or '0-125M_40hr'
% possible data types:
% 'allTrips_withDistDependence' or 'shortTrips_withDistDependence' or 'veryShortTrips'
datatype = 'veryShortTrips'; 
[runProps,turnProps] = GetSegProps_multipleDataTypes(condOI, datatype);

% [runProps,turnProps] = GetSegProps(condOI);

% [runProps, turnPropsCW, turnPropsAntiCW, ~] = ...
%     GetSegProps_v2(condOI);
% Overwrite data (if desired/for exploration)
% pCWfunc = @(t) 0.8; 

% turnProps.Plog10rad_aFunc = @(t) 1.7;
% turnProps.Plog10rad_locFunc = @(t) 0;
% turnProps.Plog10rad_scaleFunc = @(t) 0.18;
% % turnProps.Plog10rad_aFunc = @(t) 5.7;
% % turnProps.Plog10rad_locFunc = @(t) -0.26;
% % turnProps.Plog10rad_scaleFunc = @(t) 0.26;
% turnProps.pCWfunc = @(t) 0.85; 

% initial probability of run (vs turn)
initPrun = 0.5;

% parameters for initial position and direction
initdirParams.initdirType = 'random'; % 'random' or 'straightOut' or 'specified'
if strcmp(initdirParams.initdirType,'specified') % specify a desired value
    initdirParams.initTurnAngle = -0.25*pi; % -pi/4, which is 45 degrees
end


% specify filename to save
% fn2save = strcat('SimData','_',condOI,'_',datatype,'_numtrials',num2str(numtrials));
% foldername = fn2save;
% mkdir(foldername)

%% main simulation
tic
[posTrajs, headangleTrajs, segLengthTrajs, segArcProps_cell, ...
    initsegTypeVec, dispTrajs, distTrajs, totdistVec, maxdispVec, ...
    ifreturnVec, ifhitwallVec, numstepsVec] = ...
    SimTrajs_withCurvedSegs_v2(initposParams, initdirParams, initPrun, runProps, turnProps, eps, ...
    radius_arena, sensingdist, maxdist, maxNumSteps, numtrials);
toc

%% save simulation data
% q2save = {totdistVec, maxdispVec, ifreturnVec, ifhitwallVec, numstepsVec};
% qNames2save = {'totdist','maxdisp','ifreturn','ifhitwall','numsteps'};
% numq2save = length(q2save);
% for qIndx = 1:numq2save
%     writematrix(q2save{qIndx},strcat(foldername,'/',qNames2save{qIndx},'.csv'));
% end


%% Extract loops only
trialInds_loop = find(ifreturnVec==1);
totdistVec_loop = totdistVec(trialInds_loop);
maxdispVec_loop = maxdispVec(trialInds_loop);
numstepsVec_loop = numstepsVec(trialInds_loop);

%% Visualize trajectories
% close all;
% plotting parameters
numTrajs2plot = min(9,numtrials);

% choose first numTrajs2plot trajectories
trialsOI = 1:numTrajs2plot;
% randomly choose trials to plot:
% trialsOI = randi(numtrials,1,numTrajs2plot);
% choose longest loops (i.e. the longest trips that also manage to return
% [sortedDists, sortedInds] = sort(totdistVec_loop,'descend');
% trialsOI = trialInds_loop(sortedInds(1:numTrajs2plot));

posTrajsOI = posTrajs(:,:,trialsOI);
numstepsOI = numstepsVec(trialsOI);
ifhitwallOI = ifhitwallVec(trialsOI);
initsegTypeOI = initsegTypeVec(trialsOI);
segArcPropsOI = segArcProps_cell(trialsOI,:);

titlename = condOI;

VisualizeTrajectories_withCurvedSegs(posTrajsOI, segArcPropsOI,...
    initsegTypeOI,eps,numstepsOI,radius_arena,ifhitwallOI,titlename);

%% Plot distribution of trip-level properties
qMat_cell = {totdistVec,totdistVec_loop,maxdispVec,maxdispVec_loop,...
    numstepsVec,numstepsVec_loop};
qNames_cell = {'total distance','total distance (loops only)',...
    'max displacement','max displacement (loops only)','number of steps',...
    'number of steps (loops only)'};
iflog10vec = ones(1,6);
% iflog10vec(5) = 0;
ifseparatefig = false;
numbins = 50;
VisualizeDistributions(qMat_cell, qNames_cell, iflog10vec, ifseparatefig, numbins)
set(gcf,'color','w')

%% how probability of returning relate to overall properties of the trip 
% e.g. mean run length, max displacement, etc.
% close all;
quantitiesOI = {numstepsVec, maxdispVec, totdistVec};
qNames_cell = {'numsteps','maxdisp','totdist'};
numq = length(quantitiesOI);
log10quantitiesOI = cell(1,numq);
for qIndx = 1:numq
    quantityMat = quantitiesOI{qIndx};
    log10quantitiesOI{qIndx} = log10(quantityMat);
end

% iflogVec = ones(1,numq);
% iflogVec = zeros(1,numq);
iflogVec = [1,0,1];
ifexceedVec = [true,true,true];
numrows = ceil(sqrt(numq));
numcols = ceil(numq/numrows);
numbins = 100;
figure;
sgtitle(condOI);
for qIndx = 1:numq
    subplot(numrows,numcols,qIndx);
    % number of trials for different binned quantities (within trip)
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
    
    plot(binmid,PreturnVec,'-','LineWidth',1);
    % xlim([0 10]);
    xlabel(xname);
    ylabel('Preturn');
    if ifexceedVec(qIndx) == false
        title(strcat('P(return|',qNames_cell{qIndx},')'));
    else
        title(strcat('P(return|exceed ',qNames_cell{qIndx},')'));
    end

    % save data
    meanPreturnMat = [binmid;PreturnVec];
    % writematrix(meanPreturnMat,strcat(foldername,'/PreturnVS',xname,'_numbins',num2str(numbins),'.csv'));
end
set(gcf,'color','w')


%% Basic analysis and summary statistics
% fraction of trials that ended because walker encountered either food spot
% or arena boundary
numtrials_returned = sum(ifreturnVec);
numtrials_OI = numtrials_returned + sum(ifhitwallVec);
trialFrac_OI = numtrials_OI./numtrials;
disp(strcat('relevant trial fraction:',num2str(trialFrac_OI)));

% probability of encountering food spot before wall
Preturn_overall = numtrials_returned./numtrials_OI;
disp(strcat('average Preturn:',num2str(Preturn_overall)));

% average maximum displacement given that walker returns to food spot
maxDisp_condmean = sum(maxdispVec.*ifreturnVec)./numtrials_returned;

% average total distance of trip given that walker returns to food spot
totdist_condmean = sum(totdistVec.*ifreturnVec)./numtrials_returned;


%% Save data
% save(fn2save)

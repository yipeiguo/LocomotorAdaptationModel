% This is a script for comparing locomotor distributions (properties of
% runs and turns) across all groups ('allTrips','all short trips', 'very
% short trips'.

close all; clear variables;
datapathname = './';

% Specify colors
color_data = [0.5 0.5 0.5];
% color_data = [0.5 0.7 1];
% default color: [0 0.4470 0.7410]
% colormat_datatype = rand(numDataTypes,3);
% colormat_datatype = [0.3977    0.5066    0.5401; ...
%     0.9193    0.3416    0.2337; ...
%     0.7173    0.6342    0.9671];
colormat_datatype = [0.5    0.7    1; ...
    0.3693    0.0173    0.6176; ...
    0.4774    0.2772    0.2644];
% colormat_datatype = [0.8452    0.0302    0.2537; ...
%     0.3693    0.0173    0.6176; ...
%     0.4774    0.2772    0.2644];

% condition we focus on
condOI = '0-125M_24hr';

% Relationships between parameters of a distribution and statistical
% features of that distribution
% skewnorm distribution:
skewnorm_deltaFunc= @(a) a./sqrt(1+a.^2);
skewnorm_meanFunc = @(a,loc,scale) loc + scale.*skewnorm_deltaFunc(a).*sqrt(2/pi);
skewnorm_sigmaFunc = @(a,loc,scale) scale.*sqrt(1-2.*(skewnorm_deltaFunc(a).^2)./pi);
skewnorm_skewnessFunc = @(delta) (2-pi/2).*(delta.*sqrt(2/pi)).^3./(1-2.*(delta.^2)./pi).^(3/2);
skewnorm_exkurFunc = @(delta) 2.*(pi-3).*(delta.*sqrt(2/pi)).^4./(1-2.*(delta.^2)./pi).^2;

% generalized type 1 logistic distribution
genlogistic_meanFunc = @(a,loc,scale) scale.*(psi(a)-psi(1)) + loc;
genlogistic_varFunc = @(a,loc,scale) (scale.^2).*(psi(1,a)-psi(1,1));

% lognormal: % mean and sigma of log(variable)
lognormal_muFunc = @(s,loc,scale) log(scale);
lognormal_sigmaFunc = @(s,loc,scale) s;
lognormal_meanFunc_raw = @(s,loc,scale) exp(lognormal_muFunc(s,loc,scale) + ...
    lognormal_sigmaFunc(s,loc,scale).^2./2) + loc;
lognormal_varFunc_raw = @(s,loc,scale) (exp(lognormal_sigmaFunc(s,loc,scale).^2)-1).*...
    exp(2.*lognormal_muFunc(s,loc,scale) + lognormal_sigmaFunc(s,loc,scale).^2);
lognormal_meanFunc = @(mu,sigma,loc) exp(mu + sigma.^2./2) + loc;
lognormal_varFunc = @(mu,sigma) (exp(sigma.^2)-1).*exp(2.*mu + sigma.^2);


%% upload/generate data
groupNames = {'allTrips','shortTrips', 'veryShortTrips'};
numgrps = 3;
variableNames = {'log10(seg_length)','absheadturnangle','log10(effArcRadius)'};
numvars = 3;
dataCell = cell(numgrps,numvars);
for grpIndx = 1:numgrps
    grpname = groupNames{grpIndx};
    for varIndx = 1:numvars
        varname = variableNames{varIndx};
        dataCell{grpIndx,varIndx} = load(strcat(varname,'_',grpname,'_', condOI,'.csv'));
    end
end

% parameter values of fitted distributions (from jupyter notebook)
% using all trips:
% fittedparamsMat = [-2.216, 0.918, 0.737; 0.814, -0.049, 0.474];
% using only short trips:
fittedparamsCell = {[-2.216, 0.918, 0.737], [0.814, -0.05, 0.474], [1.279, 0.518, 0.248]; ...
    [-1.126, 0.455, 0.605], [0.795, -0.083, 0.656], [3.31, -0.057, 0.279];...
    [-1.101, 0.115, 0.521], [0.722, -0.188, 1.154], [2.021, -0.05, 0.191]};
qNames = {'log_{10}(run length)','|turn angle|','log_{10}(radius)'};




%% Figure
numpts = 1e3;
xminVec = [-2.2,-0.2,-1];
xmaxVec = [2,5,2.5];

figure;
for varIndx = 1:numvars
    subplot(2,2,varIndx);

    for grpIndx = 1:numgrps
        dataVec = dataCell{grpIndx,varIndx};
    
        xmin = min(dataVec); xmax = max(dataVec);
        xscan = linspace(xmin,xmax,numpts);
        fittedparams = fittedparamsCell{grpIndx,varIndx};
        if varIndx == 1 % log10(RL)
            pdfscan = skewnormal(xscan,fittedparams(1),fittedparams(2),fittedparams(3));
        elseif varIndx == 2 % turn angles
            mu = lognormal_muFunc(fittedparams(1),fittedparams(2),fittedparams(3));
            sigma = fittedparams(1);
            loc = fittedparams(2);
            Pturnangle_obj = makedist('Lognormal','mu',mu,'sigma',sigma);
            Pturnangle_obj = truncate(Pturnangle_obj,-loc,2*pi-loc);
            pdfscan = pdf(Pturnangle_obj,xscan-loc);
        elseif varIndx == 3 % radius
            pdfscan = genlogisticPDF(xscan,fittedparams(1),fittedparams(2),fittedparams(3));
        end
        histogram(dataVec,'Normalization','pdf','FaceAlpha',0.4,...
            'FaceColor',colormat_datatype(grpIndx,:),'EdgeColor', colormat_datatype(grpIndx,:));
        hold on
        plot(xscan,pdfscan,'LineWidth',1,'Color',colormat_datatype(grpIndx,:));
    end
    xlabel(qNames{varIndx});
    ylabel('P');
    xlim([xminVec(varIndx), xmaxVec(varIndx)]);
    % if varIndx == 2
    %     legend(groupNames);
    % end
end

    

















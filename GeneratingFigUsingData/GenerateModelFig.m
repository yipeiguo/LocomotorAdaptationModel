% This is a script for generating figure comparing results from run and
% turn simulations with experimental data..

close all; clear variables;
datapathname = './';

%% upload/generate data
% Load values of log10(runLengths), turnangles, log10(rad)
data_log10RL = load('log10(seg_length)_shortTrips_0-125M_24hr.csv');
data_turnangles = load('absheadturnangle_shortTrips_0-125M_24hr.csv');
data_log10rad = load('log10(effArcRadius)_shortTrips_0-125M_24hr.csv');
% data_log10RL = load('log10(seg_length)_allTrips_0-125M_24hr.csv');
% data_turnangles = load('absheadturnangle_allTrips_0-125M_24hr.csv');
% data_log10rad = load('log10(effArcRadius)_allTrips_0-125M_24hr.csv');
qNames = {'log_{10}(run length)','|turn angle|','log_{10}(radius)'};

% parameter values of fitted distributions (from jupyter notebook)
% using all trips:
% fittedparamsMat = [-2.216, 0.918, 0.737; 0.814, -0.049, 0.474];
% using only short trips:
fittedparamsMat = [-1.126, 0.455, 0.605; 0.795, -0.083, 0.656; 3.31, -0.057, 0.279];

% condition we focus on
condOI = '0-125M_24hr';
posDataTypes = {'allTrips_withDistDependence','shortTrips_withDistDependence',...
    'veryShortTrips'};
numDataTypes = length(posDataTypes);
% get run and turn properties
runProps_all = cell(1,numDataTypes);
turnProps_all = cell(1,numDataTypes);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    [runProps,turnProps] = GetSegProps_multipleDataTypes(condOI, datatype);
    runProps_all{datatypeIndx} = runProps;
    turnProps_all{datatypeIndx} = turnProps;
end


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

% Experimental data for Preturn vs variables
maxDispThresVec_data = load('maxDispThres.csv');
PreturnDataMat_givenDisp = load('PreturnMat_givenDisp.csv'); % # numx by numflies
medianPreturnData = median(PreturnDataMat_givenDisp,2,'omitnan')';
quartilesMat = quantile(PreturnDataMat_givenDisp',[0.25,0.75]);
negVec = medianPreturnData - quartilesMat(1,:);
posVec = quartilesMat(2,:) - medianPreturnData;

% Import Preturn data from simulations
meanPreturnSim_all = cell(1,numDataTypes);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    foldername = strcat(datapathname,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000));
    meanPreturnSim_all{datatypeIndx} = load(strcat(foldername,'/PreturnVSmaxdisp_numbins100.csv'));
end


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
    tripPropDataCell{propIndx} = load(strcat(fn,'.csv'));
end

% Import trip property values from simulations
tripPropNames_sim = {'numsteps','totdist','maxdisp'};
tripProp_simdata = cell(numDataTypes,numTripProps);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    foldername = strcat(datapathname,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000));
    for propIndx = 1:numTripProps
        tripProp = tripPropNames_sim{propIndx};
        tripProp_simdata{datatypeIndx,propIndx} = load(strcat(foldername,'/',tripProp,'.csv'));
    end
end


%% Plotting
close all;
F = figure;
W = 17.7;
H = 14;
set(gcf, 'PaperPositionMode','Manual', 'PaperUnits','Centimeters',...
    'PaperSize', [W H], 'PaperPosition',[0 0 W H],...
    'Units','Centimeters','Position',[5 2 W H]); 
ABCfontsize = 12;
th0 = 0.81;
th1 = 0.47; %0.82;
th2 = 0.15; % 0.33;
% annotation('textbox','Position',[0.0,th0,0.2,0.2],'String','(a)',...
%     'FontSize',ABCfontsize,'LineStyle','none');
% annotation('textbox','Position',[0.0,th1,0.2,0.2],'String','(b)',...
%     'FontSize',ABCfontsize,'LineStyle','none');
% annotation('textbox','Position',[0.0,th2,0.2,0.2],'String','(c)',...
%     'FontSize',ABCfontsize,'LineStyle','none');

plotw = 3.3;
x1 = 1.1;
x2 = x1 + plotw + 0.9;
x3 = x2 + plotw + 0.9;
x4 = x3 + plotw + 0.9;
y1 = 10.5;
y2 = 5.6;
y3 = 0.9;
% y1 = 0.9;
yVec = [y1,y2,y3];
labelfontsize = 10;
mksize = 15;

% Specify colors
color_data = [0.5 0.7 1];
% default color: [0 0.4470 0.7410]
% colormat_datatype = rand(numDataTypes,3);
% colormat_datatype = [0.3977    0.5066    0.5401; ...
%     0.9193    0.3416    0.2337; ...
%     0.7173    0.6342    0.9671];
colormat_datatype = [0.8452    0.0302    0.2537; ...
    0.3693    0.0173    0.6176; ...
    0.4774    0.2772    0.2644];

% (a) schematic to be drawn in powerpoint

% (b) Distributions of log10(runlengths), turn angles, log10(rad)
xbvec = [x2,x3,x4];
datacell = {data_log10RL,data_turnangles,data_log10rad};
numpts = 1e3;
xscan_cell = cell(1,3);
for qIndx = 1:3
    dataVec = datacell{qIndx};
    xmin = min(dataVec); xmax = max(dataVec);
    xscan = linspace(xmin,xmax,numpts);
    xscan_cell{qIndx} = xscan;
end
pdfscan_cell = cell(1,3);
% PDF for log10(RL)
pdfscan_cell{1} = skewnormal(xscan_cell{1},fittedparamsMat(1,1),...
    fittedparamsMat(1,2),fittedparamsMat(1,3));
% PDF for turnangles
mu = lognormal_muFunc(fittedparamsMat(2,1),fittedparamsMat(2,2),fittedparamsMat(2,3));
sigma = fittedparamsMat(2,1);
loc = fittedparamsMat(2,2);
Pturnangle_obj = makedist('Lognormal','mu',mu,'sigma',sigma);
Pturnangle_obj = truncate(Pturnangle_obj,-loc,2*pi-loc);
pdfscan_cell{2} = pdf(Pturnangle_obj,xscan_cell{2}-loc);
% PDF for effective radius of curvature
pdfscan_cell{3} = genlogisticPDF(xscan_cell{3},fittedparamsMat(3,1),...
    fittedparamsMat(3,2),fittedparamsMat(3,3));
xlabelposfactor = [0.1,0.23,0.2];
for qIndx = 1:3
    axes('Units','Centimeters','Position',[xbvec(qIndx), y1, plotw, plotw]);
    dataVec = datacell{qIndx};
    histogram(dataVec,'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
    hold on
    xscan = xscan_cell{qIndx};
    plot(xscan,pdfscan_cell{qIndx},'LineWidth',1,'Color',colormat_datatype(2,:));
    minx = min(xscan); maxx = max(xscan);
    miny = 0; maxy = max(pdfscan_cell{qIndx}) + 0.1;
    xlim([minx maxx]);
    ylim([miny maxy]);
    % set(gca,'XTick',[floor(minx),ceil(maxx)]); 
    set(gca,'YTick',[]); 
    % set(gca, 'box', 'off')
    text(xlabelposfactor(qIndx)*(maxx-minx)+minx,-0.2*(maxy-miny)+miny,...
        qNames{qIndx},'FontSize',labelfontsize);
    text(-0.1*(maxx-minx)+minx,0.9*(maxy-miny)+miny,'P','FontSize',labelfontsize);
end


% (c) How mean of segment properties vary with distance from food 
xCvec = [x1,x2,x3,x4];
tmax = 300;
tscan = linspace(0,tmax,1e4);
iflogx = true;
if iflogx == false
    xlabel = 'dist since food';
    xposFactor = 0.15;
    yposFactor = -0.1;
    minx = 0; maxx = tmax;
    xticks = [minx maxx];
else
    xlabel = 'log_{10}(distance)';
    xposFactor = 0.1;
    yposFactor = -0.2;
    minx = -1.5; maxx = 2.5;
    xticks = -1:1:2;
end

% Caxes = cell(1,4);
% for kk = 1:4
%     Caxes{kk} = axes('Units','Centimeters','Position',[xCvec(kk), y2, plotw, plotw]);
% end
% (c1) log10(RL)
axes('Units','Centimeters','Position',[xCvec(1), y2, plotw, plotw]);
for datatypeIndx = 1:numDataTypes
    runProps = runProps_all{datatypeIndx};
    
    % log10(RL)
    ascan = runProps.Plog10RL_aFunc(tscan);
    locscan = runProps.Plog10RL_locFunc(tscan);
    scalescan = runProps.Plog10RL_scaleFunc(tscan);
    meanlog10RLscan = skewnorm_meanFunc(ascan,locscan,scalescan).*ones(1,length(tscan));
    if iflogx == false
        plot(tscan,meanlog10RLscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    else
        plot(log10(tscan),meanlog10RLscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    end
    hold on    
end
miny = -0.3; maxy = 0.6;
xlim([minx maxx]);
ylim([miny maxy]);
set(gca,'XTick',xticks);     
text(xposFactor*(maxx-minx)+minx,yposFactor*(maxy-miny)+miny,xlabel,'FontSize',labelfontsize);
set(gca,'YTick',(miny:0.2:maxy)); 
% set(gca, 'box', 'off')
text(-0.0*(maxx-minx)+minx,1.1*(maxy-miny)+miny,'\langle log_{10}(run length) \rangle','FontSize',labelfontsize);
[lgdC1,lgdC1object] = legend({'all trips','all short trips','very short trips'});
lgdC1.Units = 'centimeters';
if iflogx == false
    % text(0.35*(maxx-minx)+minx,0.2*(maxy-miny)+miny,'(40% of short trips)','FontSize',7);
    lgdC1.Position = [xCvec(1)+0.45*plotw,y2+0.4*plotw,0.001,0.001];
else
    % text(0.1*(maxx-minx)+minx,0.6*(maxy-miny)+miny,'(40% of short trips)','FontSize',7);
    lgdC1.Position = [xCvec(1)+0.25*plotw,y2+0.8*plotw,0.001,0.001];
end
pos = lgdC1.Position;
pos(3) = lgdC1.Position(3)./2;
lgdC1.Position = pos;
lgdC1.FontUnits = 'centimeters';
lgdC1.FontSize = 8;
lgdC1.Box = 'off';
numlines = 3; % number of lines in legend
originalLegLineC1 = get(lgdC1object(numlines+1),'xdata');
leglinelengthC = originalLegLineC1(2)-originalLegLineC1(1);
for kk = 0:numlines-1
    set(lgdC1object(numlines+1+kk*2),'xdata',...
        [originalLegLineC1(1)+leglinelengthC/1.3 originalLegLineC1(2)]);
end

    
% (c2) turn angles
axes('Units','Centimeters','Position',[xCvec(2), y2, plotw, plotw]);
for datatypeIndx = 1:numDataTypes
    turnProps = turnProps_all{datatypeIndx};
    muscan = turnProps.Pturnangle_muFunc(tscan);
    sigmascan = turnProps.Pturnangle_sigmaFunc(tscan);
    locscan = turnProps.Pturnangle_locFunc(tscan);
    meanturnanglescan = lognormal_meanFunc(muscan,sigmascan,locscan).*ones(1,length(tscan));
    if iflogx == false
        plot(tscan,meanturnanglescan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    else
        plot(log10(tscan),meanturnanglescan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    end
    hold on    
end

miny = 0.4; maxy = 1.8;
xlim([minx maxx]);
ylim([miny maxy]);
set(gca,'XTick',xticks);     
text(xposFactor*(maxx-minx)+minx,yposFactor*(maxy-miny)+miny,xlabel,'FontSize',labelfontsize);
set(gca,'YTick',(miny:0.7:maxy)); 
% set(gca, 'box', 'off')
text(-0.0*(maxx-minx)+minx,1.1*(maxy-miny)+miny,'\langle |turn angle| \rangle','FontSize',labelfontsize);

% (c3) log10(rad)
axes('Units','Centimeters','Position',[xCvec(3), y2, plotw, plotw]);
for datatypeIndx = 1:numDataTypes
    turnProps = turnProps_all{datatypeIndx};
    ascan = turnProps.Plog10rad_aFunc(tscan);
    scalescan = turnProps.Plog10rad_scaleFunc(tscan);
    locscan = turnProps.Plog10rad_locFunc(tscan);
    meanlog10radscan = genlogistic_meanFunc(ascan,locscan,scalescan).*ones(1,length(tscan));   
    if iflogx == false
        plot(tscan,meanlog10radscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    else
        plot(log10(tscan),meanlog10radscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    end
    hold on    
end
miny = 0.; maxy = 0.8;
xlim([minx maxx]);
ylim([miny maxy]);
set(gca,'YTick',(miny:0.4:maxy)); 
set(gca,'XTick',xticks);     
text(xposFactor*(maxx-minx)+minx,yposFactor*(maxy-miny)+miny,xlabel,'FontSize',labelfontsize);
% set(gca, 'box', 'off')
text(-0.0*(maxx-minx)+minx,1.1*(maxy-miny)+miny,'\langle log_{10}(radius) \rangle','FontSize',labelfontsize);

% (c4) pCW
axes('Units','Centimeters','Position',[xCvec(4), y2, plotw, plotw]);
for datatypeIndx = 1:numDataTypes
    turnProps = turnProps_all{datatypeIndx};
    pCWscan = turnProps.pCWfunc(tscan).*ones(1,length(tscan));  
    if iflogx == false
        plot(tscan,pCWscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    else
        plot(log10(tscan),pCWscan,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    end
    hold on    
end
miny = 0.5; maxy = 1;
xlim([minx maxx]);
ylim([miny maxy]);
set(gca,'XTick',xticks);     
set(gca,'YTick',miny:0.1:maxy); 
text(xposFactor*(maxx-minx)+minx,yposFactor*(maxy-miny)+miny,xlabel,'FontSize',labelfontsize);
% set(gca, 'box', 'off')
text(-0.1*(maxx-minx)+minx,1.1*(maxy-miny)+miny,'P(preferred turn direction)','FontSize',labelfontsize);


% (d) compare how Preturn decays with disp between data and simulations
axes('Units','Centimeters','Position',[x1, y3, plotw, plotw]);
errorbar(log10(maxDispThresVec_data),medianPreturnData,negVec,posVec,...
    'LineStyle','none','marker','o','markerFaceColor',color_data,...
    'markerEdgeColor',color_data,'MarkerSize',4,'color',color_data);
hold on
for datatypeIndx = 1:numDataTypes
    meanPreturnMat = meanPreturnSim_all{datatypeIndx};
    plot(log10(meanPreturnMat(1,:)),meanPreturnMat(2,:),...
        'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
    hold on
end
minx = 0; maxx = 2;
miny = 0; maxy = 1;
xlim([minx maxx]);
ylim([miny maxy]);
set(gca,'XTick',[minx maxx]); 
set(gca,'YTick',miny:0.5:maxy); 
% set(gca, 'box', 'off')
text(0.1*(maxx-minx)+minx,-0.15*(maxy-miny)+miny,'log_{10}(max disp)','FontSize',labelfontsize);
text(-0.1*(maxx-minx)+minx,1.1*(maxy-miny)+miny,'P(return)','FontSize',labelfontsize);

% (e) compare trip properties between data and simulations
xEvec = [x2,x3,x4];
xlabelNames = {'log_{10}(# segments)','log_{10}(distance)','log_{10}(max disp)'};
datatypeInds2compare = [1,2,3];
numdatatypes2compare = length(datatypeInds2compare);
xminVec = [0,-1,0];
xmaxVec = [4,4,2];
ymaxVec = [2,2,4];
numbinsVec_data = [20,20,20];
numbinsVec_sim = [30,50,30];
ifSimFollowDatabin = true;
for tripPropIndx = 1:numTripProps
    axes('Units','Centimeters','Position',[xEvec(tripPropIndx), y3, plotw, plotw]);
    % experimental data:
    datavec = tripPropDataCell{tripPropIndx};
    if ifdatalogged == true
        h = histogram(datavec,numbinsVec_data(tripPropIndx),'FaceAlpha',0.5,...
            'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
    else
        h = histogram(log10(datavec),numbinsVec_data(tripPropIndx),'FaceAlpha',0.5,...
            'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
    end
    binEdges = h.BinEdges;
    binWidth = h.BinWidth;
    hold on
    for kk = 1:numdatatypes2compare
        datatypeIndx = datatypeInds2compare(kk);
        simdatavec = log10(tripProp_simdata{datatypeIndx,tripPropIndx});
        datamin = min(simdatavec);
        datamax = max(simdatavec);
        if ifSimFollowDatabin == true
            if datamin < binEdges(1)
                numexcessbins_lower = ceil((binEdges(1)-datamin)/binWidth);
            else
                numexcessbins_lower = 0;
            end
            if datamax > binEdges(end)
                numexcessbins_upper = ceil((datamax-binEdges(end))/binWidth);
            else
                numexcessbins_upper = 0;
            end
            binEdges_new = [binEdges(1) - (numexcessbins_lower:-1:1).*binWidth,...
                binEdges,binEdges(end) + (1:1:numexcessbins_upper).*binWidth];
        else
            binEdges_new = linspace(datamin,datamax,numbinsVec_sim(tripPropIndx)+1);
            binWidth = binEdges_new(2)-binEdges_new(1);
        end
        Ncounts = histcounts(simdatavec,binEdges_new);
        pdfVec = Ncounts./sum(Ncounts)/binWidth;
        binmid_new = (binEdges_new(1:end-1)+binEdges_new(2:end))./2;
        plot(binmid_new,pdfVec,'color',colormat_datatype(datatypeIndx,:),'LineWidth',1);
        % histogram(log10(simdatavec),50,'Normalization','pdf',...
        %     'FaceColor',colormat_datatype(datatypeIndx,:));
        hold on
    end
    minx = xminVec(tripPropIndx); maxx = xmaxVec(tripPropIndx);
    miny = 0; maxy = ymaxVec(tripPropIndx);
    xlim([minx maxx]);
    ylim([miny maxy]);
    set(gca,'XTick',minx:1:maxx); 
    set(gca,'YTick',[miny maxy]); 
    % set(gca, 'box', 'off')
    text(0.1*(maxx-minx)+minx,-0.18*(maxy-miny)+miny,xlabelNames{tripPropIndx},...
        'FontSize',labelfontsize);
    text(-0.1*(maxx-minx)+minx,0.9*(maxy-miny)+miny,'P','FontSize',labelfontsize);

end


%% Save Figure
% fname = sprintf('Fig_simVSdata');
% % print(gcf,[fname '.pdf'],'-dpdf','-r300');  
% % print(gcf,[fname '.png'],'-dpng','-r300');   
% saveas(gcf,fname,'epsc');




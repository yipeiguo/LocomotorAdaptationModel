% This is a script for comparing simulation and experimental data

close all; clear variables;


%% Define parameters of interest
condOI = 'Gr43a-40hr';
pathname_exptData = strcat('./',condOI,'_exptdata/');
pathname_simData = './';

posDataTypes = {'allTrips_withDistDependence','veryShortTrips'};
versions = {[],[]};
% posDataTypes = {'allTrips_withDistDependence','veryShortTrips','veryShortTrips','veryShortTrips'};
% versions = {[],[],2,3};
numDataTypes = length(posDataTypes);

dataTypeNames = {'all trips','very short trips'};

%% Extract corresponding experimental and simulation data
% Experimental data for Preturn vs variables
maxDispThresVec_data = load(strcat(pathname_exptData,'maxDispThres',condOI,'.csv'));
PreturnDataMat_givenDisp = load(strcat(pathname_exptData,'PreturnMat_givenDisp',condOI,'.csv')); % # numx by numflies
medianPreturnData = median(PreturnDataMat_givenDisp,2,'omitnan')';
quartilesMat = quantile(PreturnDataMat_givenDisp',[0.25,0.75]);
negVec = medianPreturnData - quartilesMat(1,:);
posVec = quartilesMat(2,:) - medianPreturnData;

% Import Preturn data from simulations
meanPreturnSim_all = cell(1,numDataTypes);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    foldername = strcat(pathname_simData,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000));
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
        fn = strcat(pathname_exptData,'log10(',tripProp,')_FlyData_',condOI);
    else
        fn = strcat(pathname_exptData,tripProp,'_FlyData_',condOI);
    end
    tripPropDataCell{propIndx} = load(strcat(fn,'.csv'));
end

% Import trip property values from simulations
tripPropNames_sim = {'numsteps','totdist','maxdisp'};
tripProp_simdata = cell(numDataTypes,numTripProps);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    if isempty(versions{datatypeIndx})
        foldername = strcat(pathname_simData,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000));
    else
        foldername = strcat(pathname_simData,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000),'_v',num2str(versions{datatypeIndx}));
    end
    for propIndx = 1:numTripProps
        tripProp = tripPropNames_sim{propIndx};
        tripProp_simdata{datatypeIndx,propIndx} = load(strcat(foldername,'/',tripProp,'.csv'));
    end
end

%% Plot
close all;
color_data = [0.5 0.7 1];
colormat_datatype = [0.8452    0.0302    0.2537; ...
    0.3693    0.0173    0.6176; ...
    0.4774    0.2772    0.2644; ...
    0.1560    0.8016    0.1235];

figure;
% compare how Preturn decays with disp between data and simulations
subplot(2,2,1) 
errorbar(log10(maxDispThresVec_data),medianPreturnData,negVec,posVec,...
    'LineStyle','none','marker','o','markerFaceColor',color_data,...
    'markerEdgeColor',color_data,'MarkerSize',4,'color',color_data,...
    'DisplayName','data');
hold on
for datatypeIndx = 1:numDataTypes
    meanPreturnMat = meanPreturnSim_all{datatypeIndx};
    plot(log10(meanPreturnMat(1,:)),meanPreturnMat(2,:),...
        'color',colormat_datatype(datatypeIndx,:),'LineWidth',1,...
        'DisplayName',dataTypeNames{datatypeIndx});
    hold on
end
xlabel('log_{10}(max disp)');
ylabel('P(return)');
legend('location','best');

% compare trip properties
xlabelNames = {'log_{10}(# segments)','log_{10}(distance)','log_{10}(max disp)'};
datatypeInds2compare = 1:numDataTypes; %[1,2,3,4]
numdatatypes2compare = length(datatypeInds2compare);
% xminVec = [0,-1,0];
% xmaxVec = [4,4,2];
% ymaxVec = [2,2,4];
numbinsVec_data = [20,20,20];
numbinsVec_sim = [30,50,30];
ifSimFollowDatabin = true;
for tripPropIndx = 1:numTripProps
    subplot(2,2,tripPropIndx+1); 
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
    xlabel(xlabelNames{tripPropIndx});
    ylabel('P');
    % minx = xminVec(tripPropIndx); maxx = xmaxVec(tripPropIndx);
    % miny = 0; maxy = ymaxVec(tripPropIndx);
    % xlim([minx maxx]);
    % ylim([miny maxy]);
    % set(gca,'XTick',minx:1:maxx); 
    % set(gca,'YTick',[miny maxy]); 
    % % set(gca, 'box', 'off')
    % text(0.1*(maxx-minx)+minx,-0.18*(maxy-miny)+miny,xlabelNames{tripPropIndx},...
    %     'FontSize',labelfontsize);
    % text(-0.1*(maxx-minx)+minx,0.9*(maxy-miny)+miny,'P','FontSize',labelfontsize);

end

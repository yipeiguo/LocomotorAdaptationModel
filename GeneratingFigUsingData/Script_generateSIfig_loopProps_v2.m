% This is a script for comparing loop properties 
% In this v2, we also include results from the interpolated model

close all; 
clear variables;
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

% Extract interpolated models
interpFolderName = 'SimData_0-125M_24hr_interpType_exponential_numtrials10000';
interpFuncType = 'exponential';
% Extract foldernames in main folder
d = dir(strcat(datapathname,interpFolderName));
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

% Extract values of alpha parameter from foldernames
numalphaVals = length(nameFolds);
alphascan = zeros(1,numalphaVals);
for alphaIndx = 1:numalphaVals
    fn = nameFolds{alphaIndx};
    fn = strrep(fn,'pt','.');
    alphascan(alphaIndx) = str2double(erase(fn,'alpha'));
end

colormat_interpsets = summer(ceil(numalphaVals*1.2));
optalphaIndx = 4;
colormat_datatype = [colormat_datatype;colormat_interpsets(optalphaIndx,:)];

%% Import data

% Experimental data for distributions of trip properties
tripPropNames = {'numSegs_toFoodOrBorder','distance_toFoodOrBorder','max_disp','ifloop'};
numTripProps_data = length(tripPropNames);
tripPropDataCell = cell(1,numTripProps_data);
ifdatalogged = false;
for propIndx = 1:numTripProps_data
    tripProp = tripPropNames{propIndx};
    if ifdatalogged == true
        fn = strcat('log10(',tripProp,')_FlyData_0-125M_24hr');
    else
        fn = strcat(tripProp,'_FlyData_0-125M_24hr');
    end
    tripPropDataCell{propIndx} = load(strcat(fn,'.csv'));
end

% Import trip property values from simulations
% condition we focus on
condOI = '0-125M_24hr';
posDataTypes = {'allTrips_withDistDependence','shortTrips_withDistDependence',...
    'veryShortTrips'};
numDataTypes = length(posDataTypes);

tripPropNames_sim = {'numsteps','totdist','maxdisp','ifreturn','ifhitwall'};
numTripProps_sim = length(tripPropNames_sim);
tripProp_simdata = cell(numDataTypes,numTripProps_sim);
for datatypeIndx = 1:numDataTypes
    datatype = posDataTypes{datatypeIndx};
    foldername = strcat(datapathname,'SimData','_',condOI,'_',datatype,'_numtrials',num2str(10000));
    for propIndx = 1:numTripProps_sim
        tripProp = tripPropNames_sim{propIndx};
        tripProp_simdata{datatypeIndx,propIndx} = load(strcat(foldername,'/',tripProp,'.csv'));
    end
end
tripProp_interpsets = cell(numalphaVals,numTripProps_sim);
for alphaIndx = 1:numalphaVals
    fn_currset = nameFolds{alphaIndx};
    foldername = strcat(datapathname,interpFolderName,'/',fn_currset);
    for propIndx = 1:numTripProps_sim
        tripProp = tripPropNames_sim{propIndx};
        tripProp_interpsets{alphaIndx,propIndx} = load(strcat(foldername,'/',tripProp,'.csv'));
    end
end


%% Compare loop properties
xlabelNames = {'log_{10}(# segments)','log_{10}(distance)','log_{10}(max disp)'};
ifSimFollowDatabin = true;
numbinsVec_data = [50,50,50];
xminVec = [0,-1,0];
xmaxVec = [4,4,2.1];
ymaxVec = [1.5,1,1.6];

figure;
sgtitle('Distribution of loop properties')
for tripPropIndx = 1:3
    subplot(2,2,tripPropIndx);
    % experimental data:
    datavec = tripPropDataCell{tripPropIndx};
    ifloopVec = tripPropDataCell{4};
    datavec = datavec(ifloopVec==1);
    if ifdatalogged == true
        h = histogram(datavec,'FaceAlpha',0.5,...
            'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
        % h = histogram(datavec,numbinsVec_data(tripPropIndx),'FaceAlpha',0.5,...
        %     'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
    else
        h = histogram(log10(datavec),'FaceAlpha',0.5,...
            'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
        % h = histogram(log10(datavec),numbinsVec_data(tripPropIndx),'FaceAlpha',0.5,...
        %     'Normalization','pdf','FaceColor',color_data,'EdgeColor', color_data);
    end
    binEdges = h.BinEdges;
    binWidth = h.BinWidth;
    hold on
    for datatypeIndx = 1:numDataTypes+1
        if datatypeIndx <= numDataTypes
            ifreturnVec = tripProp_simdata{datatypeIndx,4};
            simdatavec = tripProp_simdata{datatypeIndx,tripPropIndx};
        else
            ifreturnVec = tripProp_interpsets{optalphaIndx,4};
            simdatavec = tripProp_interpsets{optalphaIndx,tripPropIndx};
        end
        simdatavec = log10(simdatavec(ifreturnVec==1));
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
    
    xlabel(xlabelNames{tripPropIndx});
    ylabel('P');
end

%% Save figure
% fname = sprintf('SIFig_PloopProp_v3withInterpModel');
% print(gcf,[fname '.pdf'],'-dpdf','-r300');  
% % print(gcf,[fname '.png'],'-dpng','-r300');   
% saveas(gcf,fname,'epsc');
% 

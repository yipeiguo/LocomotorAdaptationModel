% Function for visualizing trajectories that compose of run and turn
% segments.
% The run segments are assumed to be straight lines, while the turn
% segments are circular arcs.

function VisualizeTrajectories_withCurvedSegs(posTrajs, segArcProps_cell,...
    initsegTypeVec,eps,numstepsVec,radius_arena,ifhitwallVec,titlename)

    % Extract relevant quantities
    numTrajs = size(posTrajs,3);
    numcols = ceil(sqrt(numTrajs));
    numrows = ceil(numTrajs/numcols);
    
    anglescan = linspace(-pi,pi,1e3);
    food_xscan = eps.*cos(anglescan);
    food_yscan = eps.*sin(anglescan);
    if exist('ifhitwallVec', 'var')
        assert(exist('radius_arena','var'),'radius_arena variable not provided!');
        arena_xscan = radius_arena.*cos(anglescan);
        arena_yscan = radius_arena.*sin(anglescan);
    end
        
    figure;
    if exist('titlename', 'var')
        sgtitle(titlename);
    end
    for TrajIndx = 1:numTrajs
        numsteps = numstepsVec(TrajIndx);
        posTraj = posTrajs(:,1:numsteps+1,TrajIndx);
        xmax = max(posTraj(1,:)); xmin = min(posTraj(1,:));
        ymax = max(posTraj(2,:)); ymin = min(posTraj(2,:));
        
        subplot(numrows,numcols,TrajIndx);
        plot(food_xscan,food_yscan,'k');
        hold on
        scatter(posTraj(1,1),posTraj(2,1),'ro','filled'); % exit point
        hold on
        segtype = initsegTypeVec(TrajIndx);
        for segIndx = 1:numsteps
            if segtype == 0
                plot(posTraj(1,segIndx:segIndx+1),posTraj(2,segIndx:segIndx+1),'x-');
            elseif segtype == 1
                arcProps = segArcProps_cell{TrajIndx,segIndx};
                radius = arcProps.radius;
                circlecenter = arcProps.circlecenter;
                angle_center2start = arcProps.angle_center2start;
                angle_center2end = arcProps.angle_center2end;
                turndir = arcProps.turndir;
                if turndir == 1
                    if angle_center2end < angle_center2start
                        angle_center2end = angle_center2end + 2*pi;
                    end
                elseif turndir == -1
                    if angle_center2end > angle_center2start
                        angle_center2end = angle_center2end - 2*pi;
                    end
                end
                thetarange = abs(angle_center2start - angle_center2end);
                numpts = min(max(ceil(thetarange/0.2),10),30);
                thetascan = linspace(angle_center2start,angle_center2end,numpts);
                coords_scan = circlecenter + radius.*[cos(thetascan);sin(thetascan)];
                plot(coords_scan(1,:),coords_scan(2,:),'k-','LineWidth',1);
            end
            hold on
            segtype = mod(segtype + 1,2);
        end
        % plot arena boundary if trajectory hits wall:
        if exist('ifhitwallVec', 'var')
            ifhitwall = ifhitwallVec(TrajIndx);
            if ifhitwall
                plot(arena_xscan,arena_yscan,'b');
            end
        end
        xlabel('x');
        ylabel('y');
        title(strcat('traj:',num2str(TrajIndx)));
        axis equal
        % if (xmax-xmin < eps) && (ymax-ymin < eps)
        %     xlim([xmin-1e-3,xmax+1e-3]);
        %     ylim([ymin-1e-3,ymax+1e-3]);
        % else
        %     axis equal
        % end
    end
    
end



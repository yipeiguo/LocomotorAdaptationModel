% This function for extracting potential intersection point between a
% curved circular arc and a circle.
% useful ref:
% https://math.stackexchange.com/questions/1876407/is-there-a-simpler-way-to-intersect-a-circle-and-an-arc

% Here, it is assumed that the start point of the circular arc is outside
% of the circle. We ask whether the end point of the arc intersects the
% circle.

% Inputs:
% - arcProps: structure containing the following fields: 
% ----- radius: radius of curvature of arc
% ----- circlecenter: center of circle which the arc is part of
% ----- angle_center2start: (scalar, between -pi and pi) angle of vector
% going from center of arc circle to start point of arc
% ----- angle_center2end: (scalar, between -pi and pi) angle of vector
% going from center of arc circle to end point of arc
% ----- turndir: 1 if arc goes antiCW; -1 if arc goes CW
% ----- turnangle: absolute value of turn angle of arc
% - circle_center: (2 x 1 array) coordinates of center of circle
% - circle_radius: (scalar) radius of circle
% - ifplot: (true or false) whether to plot circle, line, and potential
% intersection point.

% Outputs:
% - intersects: (true or false) whether the arc intersects circumference
% of circle
% - intersection_point: (2 x 1 array) coordinates of the (first) 
% intersection point (if it exists)
% - dist2intersect: (scalar) new arc length (from start point to
% intersection point)
% - arcProps_new: structure with properties of new arc.


function [intersects, intersection_point, dist2intersect, arcProps_new] = ...
    checkIntersection_circularArc(arcProps, circle_center, circle_radius, ifplot)

    % extract arc properties
    arcRadius = arcProps.radius;
    circlecenter_arc = arcProps.circlecenter;
    angle_center2start = arcProps.angle_center2start;
    angle_center2end = arcProps.angle_center2end;
    turndir = arcProps.turndir;
    turnangle = arcProps.turnangle;
    
    % for there to be a chance that the arc intersects the circle, the
    % distance between the two circles must be less than the sum of their
    % radii.
    distbetweenCenters = norm(circlecenter_arc - circle_center);
    canintersect = (distbetweenCenters <= (arcRadius + circle_radius)) && ...
        (distbetweenCenters >= (arcRadius - circle_radius));

    if canintersect == true
        % angle from arc circle center to reference circle center
        vec_arcCenter2refcenter = circle_center - circlecenter_arc;
        angle_arcCenter2refcenter = atan2(vec_arcCenter2refcenter(2),vec_arcCenter2refcenter(1));

        % angle of sector of arc circle between ref circle center and point
        % of intersection.
        gamma = acos((distbetweenCenters^2 + arcRadius^2 - circle_radius^2)/...
            (2*distbetweenCenters*arcRadius));

        % angles from arc circle center to the two intersection points
        if turndir == 1
            intersectangle = RenormAngle(angle_arcCenter2refcenter-gamma);
            turnangle_start2intersect = intersectangle - angle_center2start;
            if turnangle_start2intersect < 0
                turnangle_start2intersect = turnangle_start2intersect + 2*pi;
            end
            intersects = (turnangle_start2intersect <= turnangle);
            if intersects == true
                turnangle_new = turnangle_start2intersect;
            end

            % if angle_center2end > angle_center2start 
            %     intersects = (intersectangle <= angle_center2end);
            %     if intersects == true 
            %         turnangle_new = (intersectangle - angle_center2start);
            %     end
            % else
            %     if intersectangle > 0
            %         intersects = (intersectangle > angle_center2start);
            %         if intersects == true
            %             turnangle_new = intersectangle - angle_center2start;
            %         end
            %     else
            %         intersects = (intersectangle <= angle_center2end);
            %         if intersects == true
            %             turnangle_new = turnangle - (angle_center2end-intersectangle);
            %         end
            %     end
            % end
            
        elseif turndir == -1
            intersectangle = RenormAngle(angle_arcCenter2refcenter+gamma);
            turnangle_start2intersect = angle_center2start - intersectangle;
            if turnangle_start2intersect < 0
                turnangle_start2intersect = turnangle_start2intersect + 2*pi;
            end
            intersects = (turnangle_start2intersect <= turnangle);
            if intersects == true
                turnangle_new = turnangle_start2intersect;
            end

            % if angle_center2end < angle_center2start 
            %     intersects = (intersectangle >= angle_center2end);
            %     if intersects == true
            %         turnangle_new = (angle_center2start - intersectangle);
            %     end
            % else
            %     if intersectangle > 0
            %         intersects = (intersectangle >= angle_center2end);
            %         if intersects == true
            %             turnangle_new = turnangle - (intersectangle - angle_center2end);
            %         end
            %     else
            %         intersects = (intersectangle <= angle_center2start);
            %         if intersects == true
            %             turnangle_new = (angle_center2start - intersectangle);
            %         end
            % 
            %     end
            % end            
        end
    else
        intersects = false;        
    end

    if intersects == true
        intersection_point = circlecenter_arc + arcRadius.*[cos(intersectangle);sin(intersectangle)];
        dist2intersect = arcRadius*turnangle_new;
        arcProps_new = arcProps;
        arcProps_new.angle_center2end = intersectangle;
        arcProps_new.turnangle = turnangle_new;
    else
        intersection_point = [];
        dist2intersect = []; % arcRadius*turnangle;
        arcProps_new = arcProps;
    end

    % Visualize circle and arc if desired
    if ifplot == true
        figure;
        % Plot the circle
        theta = linspace(0, 2*pi, 100);
        circle_x = circle_center(1) + circle_radius * cos(theta);
        circle_y = circle_center(2) + circle_radius * sin(theta);
        plot(circle_x, circle_y, 'b', 'LineWidth', 2);
        hold on

        % plot arc
        if turndir == 1
            if angle_center2end < angle_center2start
                angle_center2end = angle_center2end + 2*pi;
            end
        elseif turndir == -1
            if angle_center2end > angle_center2start
                angle_center2end = angle_center2end - 2*pi;
            end
        end    
        thetascan_arc = linspace(angle_center2start,angle_center2end,100);
        coords_scan = circlecenter_arc + arcRadius.*[cos(thetascan_arc);sin(thetascan_arc)];
        plot(coords_scan(1,:),coords_scan(2,:),'k-','LineWidth',1);
        hold on

        % Plot the start point
        plot(coords_scan(1,1), coords_scan(2,1), ...
            'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold on

        % mark center of circular arc
        plot(circlecenter_arc(1),circlecenter_arc(2),'kx');
        hold on
    
        % Plot the intersection point if it exists
        if intersects
            plot(intersection_point(1), intersection_point(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        end
        xlabel('X');
        ylabel('Y');
        title('Circle, arc, and Intersection Point');
        axis equal;
        grid on;
    
    end

end


% Function for renormalizing angle to between -pi and pi
function angle_renorm = RenormAngle(angle)
    if angle > pi
        angle_renorm = angle - 2*pi;
    elseif angle <= -pi
        angle_renorm = angle + 2*pi;
    else
        angle_renorm = angle;
    end

end



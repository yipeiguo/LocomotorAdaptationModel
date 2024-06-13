% This is a function for extracting potential intersection point between a
% straight line and a circle.
% Here the start point of the line is assumed to be outside the circle.

% Inputs:
% - start_point: (2 x 1 array) start point of line
% - length: (scalar) length of straight line segment
% - angle: (scalar between -pi and pi) direction of travel from start point
% to end point of line. Angle is relative to x-axis
% - circle_center: (2 x 1 array) coordinates of center of circle
% - circle_radius: (scalar) radius of circle
% - ifplot: (true or false) whether to plot circle, line, and potential
% intersection point.

% Outputs:
% - intersects: (true or false) whether the line intersects circumference
% of circle
% - intersection_point: (2 x 1 array) coordinates of the (first) 
% intersection point (if it exists)
% - intersection_distance: (scalar) distance from start_point of line to
% intersection point.


function [intersects, intersection_point, intersection_distance] = ...
    checkIntersection(start_point, length, angle, circle_center, circle_radius, ifplot)

    % Calculate endpoint of the line
    end_point = start_point + length * [cos(angle); sin(angle)];

    % Calculate vector representing the line
    line_vector = end_point - start_point;

    % Vector from start_point to circle_center
    circle_vector = circle_center - start_point;

    % Project circle_vector onto line_vector
    projection = dot(circle_vector, line_vector) / norm(line_vector)^2 * line_vector;

    if dot(projection,line_vector) > 0
        % Calculate the distance between circle_center and the projected point on the line
        distance = norm(circle_vector - projection);
    
        % Check if the distance is less than or equal to the radius
        % The gives the possibility of an intersection
        canintersect = distance <= circle_radius;
    else
        canintersect = false;
    end

    % Calculate the intersection point
    if canintersect
        % Distance between projected point and the intersection point
        distance_intersect2projection = sqrt(circle_radius^2 - distance^2);
        % Distance between start point and intersection point
        intersection_distance = norm(projection) - distance_intersect2projection;

        if intersection_distance <= length
            intersects = true;
            % Calculate the intersection point
            intersection_point = start_point + intersection_distance / length * line_vector;
        else
            intersects = false;
            intersection_point = [];
        end
    else
        intersects = false;
        intersection_point = [];
        intersection_distance = [];
    end

    % Plot the circle, line, and intersection point if desired
    if ifplot == true
        figure;
        hold on;
    
        % Plot the circle
        theta = linspace(0, 2*pi, 100);
        circle_x = circle_center(1) + circle_radius * cos(theta);
        circle_y = circle_center(2) + circle_radius * sin(theta);
        plot(circle_x, circle_y, 'b', 'LineWidth', 2);
    
        % Plot the line
        plot([start_point(1), end_point(1)], [start_point(2), end_point(2)], 'r', 'LineWidth', 2);
    
        % Plot the start point
        plot(start_point(1), start_point(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
        % Plot the intersection point if it exists
        if intersects
            plot(intersection_point(1), intersection_point(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        end
    
        xlabel('X');
        ylabel('Y');
        title('Circle, Line, and Intersection Point');
        axis equal;
        grid on;
    end
end


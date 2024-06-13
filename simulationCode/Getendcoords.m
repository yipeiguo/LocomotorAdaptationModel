% This is a function that computes the coordinates at the end of a segment
% given the starting coordinates, the starting heading angle, the run
% length and change in heading angle during the segment.

% inputs:
% - coords_start: (2x1 array) with x and y coordinates at start of segment
% - turnangle: (scalar between 0 and 2*pi) unsigned
% - turndir: (+1 or -1) +1 if antiCW; -1 if CW

function [coords_end, headangle_end, arcProps] = Getendcoords(coords_start, ...
    headangle_start, seglength, turnangle, turndir)

    % radius of circular arc (unsigned)
    radius = seglength/turnangle;

    % coordinates of center
    [circlecenter, angle_start2center] = ...
        Getcenter(coords_start, headangle_start, radius, turndir);
    % angle of vector from center to starting position:
    angle_center2start = RenormAngle(angle_start2center - turndir*pi);

    % angle from center to final position
    angle_center2end = RenormAngle(angle_center2start + turndir*turnangle);

    % net displacement vector
    coords_end = circlecenter + radius.*[cos(angle_center2end);sin(angle_center2end)];

    % final heading angle
    headangle_end = RenormAngle(headangle_start + turndir*turnangle);

    % store properties of circular arc
    arcProps.radius = radius;
    arcProps.circlecenter = circlecenter;
    arcProps.angle_center2start = angle_center2start;
    arcProps.angle_center2end = angle_center2end;
    

end

% Function for computing coordinates of center of circle of arc:
function [circlecenter, angle] = Getcenter(coords_start, ...
    headangle_start, radius, turndir)

    angle = RenormAngle(headangle_start + turndir*pi/2);

    circlecenter = coords_start + radius.*[cos(angle);sin(angle)];

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

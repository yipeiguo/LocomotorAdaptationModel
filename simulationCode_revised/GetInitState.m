% This is a function for drawing or specifying initial position and 
% direction of walker.

% This is a function for drawing the exit point (at the circumference of 
% food spot) and the initial heading angle of the agent.

% outputs:
% - exitpt: (2x1 array) coordinates of exit point
% - dir_init: (scalar) initial heading direction (wrt x-axis) at exitpt
% - initTurnAngle: (scalar) angle between radial vector from origin and the
% heading angle

function [exitpt, dir_init, initTurnAngle] = ...
    GetInitState(initposParams, initdirParams)

    
    initdirType = initdirParams.initdirType;
    eps = initposParams.eps; % food patch radius
    
    % point of exit from food patch
    exitptAngle = -pi + rand*2*pi;
    exitpt = eps.*[cos(exitptAngle);sin(exitptAngle)];
    
    % draw travelling direction (between -pi/2 and pi/2) relative to the boundary 
    % of the food spot. Think of this as the turn angle at the boundary
    % assuming that the animal was travelling straight form the origin to
    % the boundary (antiCW turn if initTurnAngle > 0)
    if strcmp(initdirType,'straightOut') % straight from food spot
        initTurnAngle = 0;
    elseif strcmp(initdirType,'random') % random directions with equal probabilities
        initTurnAngle = -pi/2 + rand(1)*pi; 
    elseif strcmp(initdirType,'specified') % specify a desired value
        initTurnAngle = initdirParams.initTurnAngle;
    end
    
    % new angle (upon exit)
    dir_init = exitptAngle + initTurnAngle;
    if dir_init > pi
        dir_init = dir_init - 2*pi;
    elseif dir_init <= - pi
        dir_init = dir_init + 2*pi;
    end


    
end


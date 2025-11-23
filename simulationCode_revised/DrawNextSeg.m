% This is a function for drawing next step of the simulation.

% Inputs:
% - pos_curr: [2 x 1 vector] representing [x;y] coordinates of current
% position
% - dir_curr: (scalar between -pi and pi) current heading direction with 
% 0 representing positive x-axis
% - cumdist: (scalar) quantity used to regulate locomotor parameters
% - segProps: structure containing the following fields:
% --- segtype: 'run' or 'turn'
% --- for 'run': 'Plog10RL_aFunc','Plog10RL_locFunc','Plog10RL_scaleFunc'
% --- for 'turn': 
%          'Pturnangle_muFunc','Pturnangle_sigmaFunc','Pturnangle_locFunc',
%          'Plog10rad_aFunc','Plog10rad_locFunc','Plog10rad_scaleFunc',
%          'pCWfunc'

% Run segments are assumed to be straight lines with log10(runlength)
% drawn from a skewnormal distribution.
% Turn segments are assumed to be circular arcs with log10(radius of 
% curvature) drawn from a type 1 generalized logistic distribution, and the
% turn angle drawn from a (truncated) lognormal distribution.

% Outputs:
% - turndir: (scalar) 1 if antiCW, -1 if CW

function [pos_new, dir_new, seglength, arcProps] = ...
    DrawNextSeg(pos_curr, dir_curr, cumdist, segProps)

    % extract segment type
    segtype = segProps.segtype;

    if strcmp(segtype,'run')

        % extract run properties, log10(RL) assumed to be drawn from skew
        % normal distribution
        Plog10RL_aFunc = segProps.Plog10RL_aFunc;
        Plog10RL_locFunc = segProps.Plog10RL_locFunc;
        Plog10RL_scaleFunc = segProps.Plog10RL_scaleFunc;

        % sample next run length
        Plog10RL_a = Plog10RL_aFunc(cumdist);
        Plog10RL_loc = Plog10RL_locFunc(cumdist);
        Plog10RL_scale = Plog10RL_scaleFunc(cumdist);
        log10RL = skewnormalrnd(Plog10RL_a,Plog10RL_loc,Plog10RL_scale,1);
        seglength = 10^log10RL;

        % update position
        pos_new = pos_curr + seglength.*[cos(dir_curr);sin(dir_curr)];

        % heading direction remains unchanged:
        dir_new = dir_curr;

        arcProps = [];

    elseif strcmp(segtype,'turn')

        % Extract turn properties
        % turn angles are assumed to be drawn from truncated lognormal
        % distribution with parameters mu and sigma (and shifted by loc).
        Pturnangle_muFunc = segProps.Pturnangle_muFunc;
        Pturnangle_sigmaFunc = segProps.Pturnangle_sigmaFunc;
        Pturnangle_locFunc = segProps.Pturnangle_locFunc;
        % log10(radius of curvature) assumed to be drawn from generalized
        % logistic distribution        
        Plog10rad_aFunc = segProps.Plog10rad_aFunc;
        Plog10rad_locFunc = segProps.Plog10rad_locFunc;
        Plog10rad_scaleFunc = segProps.Plog10rad_scaleFunc;        
        % how probability of turning in preferred direction changes over trip:
        pCWfunc = segProps.pCWfunc;

        % Draw radius of curvature
        Plog10rad_a = Plog10rad_aFunc(cumdist);
        Plog10rad_loc = Plog10rad_locFunc(cumdist);
        Plog10rad_scale = Plog10rad_scaleFunc(cumdist);
        log10radius = genlogisticrnd(Plog10rad_a,Plog10rad_loc,Plog10rad_scale,1);
        radius = 10^log10radius;
        
        % Draw turn direction (CW or antiCW)
        pCW = pCWfunc(cumdist);
        turndir = sign(rand - pCW); % +1 if antiCW; -1 if CW
    
        % draw unsigned turn angle
        Pturn_mu = Pturnangle_muFunc(cumdist);
        Pturn_sigma = Pturnangle_sigmaFunc(cumdist);
        Pturn_loc = Pturnangle_locFunc(cumdist);
        Pturnangle_obj = makedist('Lognormal','mu',Pturn_mu,'sigma',Pturn_sigma);
        Pturnangle_obj = truncate(Pturnangle_obj,-Pturn_loc,2*pi-Pturn_loc);
        turnangle = random(Pturnangle_obj) + Pturn_loc;

        % segment length: l = r*theta
        seglength = radius*turnangle;

        % update position and heading angle
        [pos_new, dir_new, arcProps] = Getendcoords(pos_curr, ...
            dir_curr, seglength, turnangle, turndir);
        arcProps.turndir = turndir;
        arcProps.turnangle = turnangle;
    end

    
end
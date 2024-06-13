% This is a function for simulating trajectories based on distributions
% of straight-line runs and curved turn segments extracted from the data.

% In this v2, we allow the agent to have some sensing radius, such that it
% is considered to have returned if it comes within that distance of the
% food spot after having left that region.


% Inputs:
% - eps: radius of food spot

function [posTrajs, headangleTrajs, segLengthTrajs, segArcProps_cell, ...
    initsegTypeVec, dispTrajs, distTrajs, totdistVec, maxdispVec, ...
    ifreturnVec, ifhitwallVec, numstepsVec] = SimTrajs_withCurvedSegs_v2(...
    initposParams, initdirParams, initPrun, runProps, turnProps, eps, ...
    radius_arena, sensingdist, maxdist, maxNumSteps, numtrials)

    % storage arrays
    initsegTypeVec = ones(1,numtrials).*10; % initial segmenttype (0 for runs, 1 for turns)
    posTrajs = zeros(2,maxNumSteps+1,numtrials); % position at start and end of each segment
    headangleTrajs = zeros(numtrials,maxNumSteps+1); % heading angle (wrt x-axis) at the start and end of each segment
    segLengthTrajs = zeros(numtrials,maxNumSteps); % step size at each step
    segArcProps_cell = cell(numtrials,maxNumSteps); % arc properties of each segment
    dispTrajs = zeros(numtrials,maxNumSteps); % displacement from origin at the end of every step
    distTrajs = zeros(numtrials,maxNumSteps); % distance travelled since leaving food spot
    totdistVec = zeros(1,numtrials);
    maxdispVec = zeros(1,numtrials);
    ifreturnVec = zeros(1,numtrials);
    ifhitwallVec = zeros(1,numtrials);
    numstepsVec = zeros(1,numtrials);
    
    for trialIndx = 1:numtrials
        if rem(trialIndx,100) == 0
            fprintf('trialIndx = %d \n', trialIndx);
        end
        % draw exit point and initial heading direction:
        % note that this doesn't count as first step
        [exitpt, dir_init, ~] = ...
            GetInitState(initposParams, initdirParams);
        pos_curr = exitpt;
        dir_curr = dir_init;
        
        % store exit point and initial heading direction
        posTrajs(:,1,trialIndx) = exitpt;
        headangleTrajs(trialIndx,1) = dir_curr;

        % draw initial segment type
        segtype = (sign(rand-initPrun)+1)/2;
        initsegTypeVec(trialIndx) = segtype;

        % initialize variables
        disp_curr = eps;
        % maxdisp_curr = eps;
        totdist_curr = 0;
        returnzoneRad = eps;
        ifreturn = false;
        ifhitwall = false;
        
        dispTrajs(trialIndx,1) = disp_curr;
        distTrajs(trialIndx,1) = totdist_curr;
        
        % errorflag = false;
        % main simulation (loop over each step)
        for stepIndx = 1:maxNumSteps
            if segtype == 0 % current segment is a run
                segProps = runProps;
                segProps.segtype = 'run';
            elseif segtype == 1 % current segment is a turn
                segProps = turnProps;
                segProps.segtype = 'turn';
            end
            % totdist_curr
            % draw next segment and update position and heading
            [pos_new, dir_new, seglength, arcProps] = ...
                DrawNextSeg(pos_curr, dir_curr, totdist_curr, segProps);
            
            % try
            %     [pos_new, dir_new, seglength, arcProps] = ...
            %         DrawNextSeg(pos_curr, dir_curr, totdist_curr, segProps);
            % catch
            %     % disp(totdist_curr)
            %     errorflag = true;
            % end
            % if errorflag == true
            %     break
            % end
            
            % for each segments, check if it enters the food spot
            if (segtype == 0) && (disp_curr > returnzoneRad)
                [intersects, intersection_point, dist2intersect] = ...
                    checkIntersection(pos_curr, seglength, dir_curr, [0;0], returnzoneRad, false);                
            elseif segtype == 1
                [intersects, intersection_point, dist2intersect, arcProps_new] = ...
                    checkIntersection_circularArc(arcProps, [0;0], returnzoneRad, false);                
            else
                intersects = false;
            end
            if intersects == true
                pos_new = intersection_point;
                seglength = dist2intersect;
                ifreturn = true;
                if segtype == 1
                    arcProps = arcProps_new;
                end
            end

            disp_new = sqrt(sum(pos_new.^2));
            % check if segment excees arena boundary
            if disp_new >= radius_arena
                if segtype == 0
                    [pos_new, seglength] = Getposboundary(pos_curr,pos_new,radius_arena);
                elseif segtype == 1
                    % flip the arc to get intersection point
                    arcProps_flipped = arcProps;
                    arcProps_flipped.angle_center2start = arcProps.angle_center2end;
                    arcProps_flipped.angle_center2end = arcProps.angle_center2start;
                    arcProps_flipped.turndir = -arcProps.turndir;
                    [intersects, intersection_point, dist2intersect, arcProps_flipped_new] = ...
                        checkIntersection_circularArc(arcProps_flipped, [0;0], radius_arena, false);
                    assert(intersects == true, 'intersection not found at arena boundary!')
                    % update final segment
                    pos_new = intersection_point;                    
                    seglength = seglength - dist2intersect;
                    arcProps.turnangle = arcProps.turnangle - arcProps_flipped_new.turnangle;
                    arcProps.angle_center2end = arcProps_flipped_new.angle_center2end;
                    dir_new = RenormAngle(dir_curr + arcProps.turndir*arcProps.turnangle);
                end
                disp_new = radius_arena;
                ifhitwall = true;
            end
            % if disp_new <= eps
            %     disp_new = eps;
            %     ifreturn = true;
            % end
            % update variables for next step
            totdist_curr = totdist_curr + seglength;
            disp_curr = disp_new;
            segtype = mod(segtype + 1,2);
            % if displacement exceeds eps + sensingdist, we increase the return zone radius
            if disp_curr > eps + sensingdist
                returnzoneRad = eps + sensingdist;
            end

            % store variables
            posTrajs(:,stepIndx+1,trialIndx) = pos_new;
            headangleTrajs(trialIndx,stepIndx+1) = dir_new;
            segLengthTrajs(trialIndx,stepIndx) = seglength;
            dispTrajs(trialIndx,stepIndx) = disp_curr;
            distTrajs(trialIndx,stepIndx) = totdist_curr;
            segArcProps_cell{trialIndx,stepIndx} = arcProps;
        
            if (totdist_curr < maxdist) && (ifreturn == false) && (ifhitwall == false)
                pos_curr = pos_new;
                dir_curr = dir_new;
                
            else
                break
            end
            
        end
        % store properties of trial
        totdistVec(trialIndx) = totdist_curr;
        maxdispVec(trialIndx) = max(dispTrajs(trialIndx,:));
        ifreturnVec(trialIndx) = ifreturn; 
        ifhitwallVec(trialIndx) = ifhitwall; 
        numstepsVec(trialIndx) = stepIndx;
    end
    
end

% Function for extracting point of contact with arena boundary (if the
% animal exceeds boundary in current time step) for straight-line runs
% Inputs:
% - pos_curr: [x1,y1] representing position of walker on current time point
% - pos_new: [x2,y2] representing newly drawn position
% - R: radius of arena
function [pos_boundary, stepsize] = Getposboundary(pos_curr,pos_new,R)
    R1sq = sum(pos_curr.^2);
    assert(R1sq < R^2, 'pos_curr must be inside arena!');
    R2sq = sum(pos_new.^2);
    assert(R2sq >= R^2, 'pos_new must be outside arena!');
    prod = sum(pos_curr.*pos_new);
    % coefficients of quadratic equation:
    a = R1sq + R2sq - 2*prod;
    b = 2*(prod - R1sq);
    c = R1sq - R^2;
    alpha = (-b + sqrt(b^2 - 4*a*c))/(2*a);
    
    % position at boundary:
    pos_boundary = (1-alpha).*pos_curr + alpha.*pos_new;
    stepsize = sqrt(sum((pos_boundary-pos_curr).^2));
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


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:07:47 2022

@author: labadmin
"""
# Note that in the v2, we use move segments rather than run segments.
# 3/31/24

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from helper_functions import rle
from statsmodels.stats.stattools import medcouple

# Function for augmenting per-frame data frame with desired columns
def AugmentPerFrameDF(df,metadata):
    time = [] # total amount of time that has passed, including the current frame
    CF = [] # cumulative feeding duration, including current frame
    CFF = [] # cumulative feeding fraction, including current frame
    dist_center = [] # distance to center of food spot
    dist_from_patch = [] # distance from nearest edge of food patch (if fly is not on patch)
    totdist = [] # total distance travelled, including current frame
    angular_pos = [] # angular position (angle in rad of the vector going from center of food spot to the body of fly)
    rel_angle = [] # angle between heading direction of fly and its angular position (angle-angular_pos)
    
    for fly in tqdm(df.fly.unique()):
        dt_rel = df[df.fly==fly].dt.values
        etho_rel = df[df.fly==fly].ethogram.values
        cumtime = np.cumsum(dt_rel)
        cumfeedtime = np.cumsum(dt_rel*(etho_rel==3))
        time.append(cumtime)
        CF.append(cumfeedtime)
        CFF.append(cumfeedtime/cumtime)
        
        # location and properties of food spot as well as arena size
        food_x = metadata[fly]['arena']['spots']['x']
        food_y = metadata[fly]['arena']['spots']['y']
        scale = metadata[fly]['arena']['scale']
        food_radius = metadata[fly]['arena']['spots']['radius'] /scale
        body_x = df[df.fly==fly].body_x.values
        body_y = df[df.fly==fly].body_y.values
        dist2center = np.sqrt((food_x-body_x)**2+(food_y-body_y)**2)
        dist2patch = dist2center - food_radius
        dist2patch[dist2patch<0] = 0
        ang2center = np.arcsin((body_y-food_y)/dist2center)
        dist_center.append(dist2center)
        dist_from_patch.append(dist2patch)
        angular_pos.append(ang2center)
        
        # total distance travelled so far (since first frame)
        dispVec = np.append(0,np.sqrt((body_x[1:]-body_x[0:-1])**2 + (body_y[1:]-body_y[0:-1])**2))
        totdist.append(np.nancumsum(dispVec))
        
        # heading angle relative to angular position
        head_angle = df[df.fly==fly].angle.values
        relative_angle = head_angle - ang2center
        relative_angle[relative_angle >= np.pi] = relative_angle[relative_angle >= np.pi] - 2*np.pi
        relative_angle[relative_angle < -np.pi] = relative_angle[relative_angle < -np.pi] + 2*np.pi
        rel_angle.append(relative_angle)
        
        
    df['time'] = np.hstack(time)
    df['CF'] = np.hstack(CF)
    df['CFF'] = np.hstack(CFF)
    df['dist_center'] = np.hstack(dist_center)
    df['dist_from_patch'] = np.hstack(dist_from_patch)
    df['totdist'] = np.hstack(totdist)
    df['angular_pos'] = np.hstack(angular_pos)
    df['rel_angle'] = np.hstack(rel_angle) 
    
    return df

# function for creating PerMoveSegment dataframe
# These movement segments can be runs or turns or both, etc. specified through 'ethogramOI'
# In this v2, the turn angles are between -2*pi and 2*pi so that we can distinguish between small turns (< pi) and big turns (>pi).
# We also indicate whether the turns are in the preferred directions
def CreatePerMoveSegmentDF_v2(df,metadata,ethogramOI):
    perMoveSegDf = pd.DataFrame()
    for condition in tqdm(df.condition.unique()):
        for fly in tqdm(df.fly.unique()):
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]

            # location and properties of food spot as well as arena size
            food_x = metadata[fly]['arena']['spots']['x']
            food_y = metadata[fly]['arena']['spots']['y']

            segL, segSt, segType = rle(flyDf.segment)

            # runDf_fly = pd.DataFrame()
            moveSegDf_fly = {'fly': [], 
                  'condition': [],
                  'seg_state': [],  
                  'etho_state': [],           
                  'after_which_visit': [],
                  'dist_since_visit': [], # total distance travelled during runs since last food spot visit
                  'time_since_visit': [], # time since last food spot visit
                  'cumRunTime_since_visit': [], # total run time since last food spot visit       
                  'moveSegIndx': [], # which movement segment since the last change in segment state (food spot visit/loop/border->food, etc.)
                  'seg_duration': [],
                  'seg_length': [], # total distance travelled by fly during this segment
                  'disp_from_center': [], # displacement of fly from center of food spot at the start of movement segment
                  'seg_disp': [], # displacement of fly during this movement segment
                  'velo': [],  # average speed of fly during this segment (seg_length/seg_duration)
                  'headturnangle': [], # net heading turn angle during movement segment (body angle at end - body angle at start)
                  'absheadturnangle': [], # absolute value of the turn angle
                  'numDirChanges': [], # changes in turn directions during segment           
                  'ifbigturn': [], # if the first direction change corresponds to a big turn (>pi) 
                  'ifCW': [], # whether the turn is in CW direction
                  'netReorientationAngle': [], # net change in angle in movement direction between the end and start of segment
                  'absReorientAngle': [], # absolute value of reorientation angle
                  'startmoveAngle_rel2food': [], # movement angle at start of segment (relative to the vector from food to current position)  
                }
            ifpreferredturn_fly = [] # For each movement segment, denote whether it is turn in the preferred direction for this segment state (e.g. food->food 
                                     # or food->border)
            numSegs_fly = [] # total number of movement segments in this segment state
            prev_visit = 0
            cumdist_curr = 0
            cumRunTime_curr = 0                    
            cumtime_curr = 0
            runfound = False
            for ii, ss in enumerate(segSt):
                segtype_curr = segType[ii]
                if segtype_curr == 1:
                    prev_visit = prev_visit + 1
                if (segtype_curr == 1) or (segtype_curr == 2) or (segtype_curr == 4):
                    cumdist_curr = 0
                    cumRunTime_curr = 0
                    cumtime_curr = 0 # reset time since leaving food spot

                se = min(ss+segL[ii], len(flyDf.body_x.values))
    
                ethoL, ethoSt, ethoType, ethoDuration = rle(flyDf.ethogram[ss:se], flyDf.dt[ss:se])
                moveSegs = np.where(np.isin(ethoType,ethogramOI))[0]
                # runSegs = np.where(ethoType==2)[0]
                etho_StartTime = np.cumsum(np.insert(ethoDuration,0,0))
                moveSegs_starttime = etho_StartTime[moveSegs]
                moveSegs_ethoType = ethoType[moveSegs]
                
                if len(moveSegs) > 0: 
                    ifCW_allsegs = []
                    for moveIndx in range(len(moveSegs)):

                        # storage for data frame
                        # fly id and condition
                        moveSegDf_fly['fly'].append(fly)
                        moveSegDf_fly['condition'].append(condition)

                        # segment state
                        moveSegDf_fly['seg_state'].append(segtype_curr)

                        # ethogram state
                        moveSegDf_fly['etho_state'].append(moveSegs_ethoType[moveIndx])

                        # number of food visits (entrance to food spot) before this movement segment
                        moveSegDf_fly['after_which_visit'].append(prev_visit)

                        # which movement segment is it since entering this segment state
                        moveSegDf_fly['moveSegIndx'].append(moveIndx)

                        # segment duration
                        movedur = ethoDuration[moveSegs[moveIndx]]
                        moveSegDf_fly['seg_duration'].append(movedur)
                        # store total running time since last visit
                        moveSegDf_fly['cumRunTime_since_visit'].append(cumRunTime_curr)
                        if segtype_curr != 1:
                            cumRunTime_curr = cumRunTime_curr + movedur
                        # store total time since last visit
                        moveSegDf_fly['time_since_visit'].append(cumtime_curr + moveSegs_starttime[moveIndx])
                        
                        
                        # segment length (distance)
                        startframe = max(ss + ethoSt[moveSegs[moveIndx]],0)
                        endframe = min(ss + ethoSt[moveSegs[moveIndx]] + ethoL[moveSegs[moveIndx]] + 1, len(flyDf.body_x.values)) # one frame after last frame
                        xpos_all = flyDf.body_x.values[startframe:endframe]
                        ypos_all = flyDf.body_y.values[startframe:endframe]
                        dist_all = np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2)
                        totdist = np.nansum(dist_all)
                        moveSegDf_fly['seg_length'].append(totdist)
                        # store total running distance since last visit
                        moveSegDf_fly['dist_since_visit'].append(cumdist_curr)                        
                        if segtype_curr != 1:
                            cumdist_curr = cumdist_curr + totdist
                            
                        # starting displacement from center of spot
                        dispVec_fromcenter = np.array([xpos_all[0]-food_x,ypos_all[0]-food_y])
                        moveSegDf_fly['disp_from_center'].append(np.sqrt(np.sum(dispVec_fromcenter**2)))

                        # segment length (displacement)
                        disp = np.sqrt((xpos_all[-1]-xpos_all[0])**2 + (ypos_all[-1]-ypos_all[0])**2)
                        moveSegDf_fly['seg_disp'].append(disp)

                        # mean velocity
                        moveSegDf_fly['velo'].append(totdist/movedur)
                        
                        # heading angle (vector from body to head of fly):
                        if endframe - startframe > 1:
                            headingAngle_all = flyDf.angle.values[startframe:endframe]
                            headturnangle_all = computeTurnAngBetween2angles(headingAngle_all[0],headingAngle_all[1:])
                            turnAngleDir_all = np.sign(headturnangle_all)
                            changedirFrameInds = np.where(turnAngleDir_all[0:-1] != turnAngleDir_all[1:])[0]
                            numchanges = len(changedirFrameInds)
                            moveSegDf_fly['numDirChanges'].append(numchanges)
                            if numchanges == 0:
                                headturnangle_currseg = headturnangle_all[-1]
                                moveSegDf_fly['ifbigturn'].append(0)
                            else:
                                # investigate the first direction change
                                changeFrameIndx = changedirFrameInds[-1] # frame right before change in direction
                                if (np.abs(headturnangle_all[changeFrameIndx]) > np.pi/2) and (np.abs(headturnangle_all[changeFrameIndx+1]) > np.pi/2):
                                    moveSegDf_fly['ifbigturn'].append(1)
                                    headturnangle_last = headturnangle_all[-1]
                                    headturnangle_currseg = -np.sign(headturnangle_last)*2*np.pi + headturnangle_last
                                else:
                                    moveSegDf_fly['ifbigturn'].append(0)
                                    headturnangle_currseg = headturnangle_all[-1]
                            moveSegDf_fly['headturnangle'].append(headturnangle_currseg)
                            moveSegDf_fly['absheadturnangle'].append(np.abs(headturnangle_currseg))
                            ifCW_currseg = headturnangle_currseg<0
                            moveSegDf_fly['ifCW'].append(ifCW_currseg)
                            ifCW_allsegs.append(ifCW_currseg)
                            
                        else:
                            moveSegDf_fly['numDirChanges'].append(0)
                            moveSegDf_fly['ifbigturn'].append(0)
                            moveSegDf_fly['headturnangle'].append(0)
                            moveSegDf_fly['absheadturnangle'].append(0)
                            moveSegDf_fly['ifCW'].append(np.nan)
                            ifCW_allsegs.append(np.nan)

                        # initial and end movement vectors (wrt x-axis)
                        # note endframe-startframe is the #frames the segment contains
                        if endframe - startframe >= 2:
                            startmoveVec = [flyDf.body_x.values[np.minimum(startframe + 1,len(flyDf)-1)]-flyDf.body_x.values[np.maximum(startframe - 1,0)],
                                            flyDf.body_y.values[np.minimum(startframe + 1,len(flyDf)-1)]-flyDf.body_y.values[np.maximum(startframe - 1,0)]]
                            endmoveVec = [flyDf.body_x.values[np.minimum(endframe,len(flyDf)-1)]-flyDf.body_x.values[np.maximum(endframe - 2,0)],
                                         flyDf.body_y.values[np.minimum(endframe,len(flyDf)-1)]-flyDf.body_y.values[np.maximum(endframe - 2,0)]]
                            angleOI = computeTurnAngBetween2vectors(startmoveVec,endmoveVec)
                            moveSegDf_fly['netReorientationAngle'].append(angleOI)
                            moveSegDf_fly['absReorientAngle'].append(np.abs(angleOI))
                        else:
                            moveSegDf_fly['netReorientationAngle'].append(0)
                            moveSegDf_fly['absReorientAngle'].append(0)
                        
                        # travelling angle (wrt food source) at the start of the movement segment
                        relangle = computeTurnAngBetween2vectors(dispVec_fromcenter,startmoveVec)
                        moveSegDf_fly['startmoveAngle_rel2food'].append(relangle)

                    # Extract preferred turn direction for this segment
                    ifCW_allsegs = np.array(ifCW_allsegs)
                    numCWturns = np.sum(ifCW_allsegs == True) #ifCW_allsegs.count(True)
                    numAntiCWturns = np.sum(ifCW_allsegs == False) #ifCW_allsegs.count(False)
                    if numCWturns > numAntiCWturns:
                        ifpreferredTurn_allsegs = np.copy(ifCW_allsegs)
                    elif numCWturns < numAntiCWturns:
                        ifpreferredTurn_allsegs = np.copy(ifCW_allsegs)
                        ifpreferredTurn_allsegs[ifCW_allsegs==1] = 0
                        ifpreferredTurn_allsegs[ifCW_allsegs==0] = 1
                    else:
                        ifpreferredTurn_allsegs = np.ones((len(moveSegs),))*0.5
                    ifpreferredturn_fly.append(ifpreferredTurn_allsegs)

                    # store the total number of movement segments in this segment state
                    numSegs_fly.append(np.ones((len(moveSegs),))*len(moveSegs))
                        
                        
                if (segtype_curr == 4) or (segtype_curr == 5) or (segtype_curr == 0):
                    cumtime_curr = cumtime_curr + etho_StartTime[-1]
           
            moveSegDf_fly = pd.DataFrame(moveSegDf_fly)
            if len(ifpreferredturn_fly) > 0: 
                moveSegDf_fly['ifpreferredturn'] = np.hstack(ifpreferredturn_fly)
                moveSegDf_fly['numSegs'] = np.hstack(numSegs_fly)

            perMoveSegDf = perMoveSegDf.append(moveSegDf_fly, sort=False)
            
    return perMoveSegDf

def CreatePerFlyDF(df):
    
    per_fly_df = {'fly': [], 
              'condition': [],
              'total_feeding': [],
              'mean_visitdur': [],
              'first_visitdur': [],
              'first_visitdur_scaled': [], # first visit duration scaled by mean visit duration of fly
              'sumfirst2visitdur': [], # sum of first two visit durations
              'mean_visitdur_after1': [],
              'mean_visitdur_after2': [],
              'num_visits': [],
              'first_hit_time': [],
              'first_hit_distance': [],
              'first_loop_time': [], # first trip from food spot
              'first_loop_distance': [],
              'CFF_afterfirstvisit':[],
            }

    for condition in tqdm(df.condition.unique()):
        for fly in tqdm(df.fly.unique()):
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]
            
            # extract relevant quantities
            dt = flyDf.dt.values
            etho = flyDf.ethogram.values
            CFF = flyDf.CFF.values
            
            # number of visits, and mean duration of visits
            segments = flyDf.segment.values
            runlen, pos, state, dur = rle(segments, dt=dt) # see function def above
            if len(dur)>0:
                visit_durations = dur[state==1]
                num_visits = len(visit_durations)
                # fly id
                per_fly_df['fly'].append(fly)
                per_fly_df['condition'].append(condition)
                per_fly_df['num_visits'].append(num_visits)
                if num_visits > 0:
                    per_fly_df['total_feeding'].append(np.sum(dt[etho==3]))
                    meanvisitdur = np.nanmean(visit_durations)
                    per_fly_df['mean_visitdur'].append(meanvisitdur)
                    firstvisitdur = visit_durations[0]
                    per_fly_df['first_visitdur'].append(firstvisitdur)
                    per_fly_df['first_visitdur_scaled'].append(firstvisitdur/meanvisitdur)
                    if num_visits > 1:
                        sumVisit1and2dur = firstvisitdur + visit_durations[1]
                    else:
                        sumVisit1and2dur = firstvisitdur
                    per_fly_df['sumfirst2visitdur'].append(sumVisit1and2dur)
                    
                    if num_visits>1:
                        per_fly_df['mean_visitdur_after1'].append(np.nanmean(visit_durations[1:]))
                        if num_visits>2:
                            per_fly_df['mean_visitdur_after2'].append(np.nanmean(visit_durations[2:]))
                        else:
                            per_fly_df['mean_visitdur_after2'].append(np.nan)
                    else:
                        per_fly_df['mean_visitdur_after1'].append(np.nan)
                        per_fly_df['mean_visitdur_after2'].append(np.nan)
                    # distance travelled before first feeding
                    frame_firstarr = pos[state==1][0]
                    xVec = flyDf.body_x.values[0:frame_firstarr]
                    yVec = flyDf.body_y.values[0:frame_firstarr]
                    initdist = np.nansum(np.sqrt((xVec[1:]-xVec[0:-1])**2 + (yVec[1:]-yVec[0:-1])**2))
                    per_fly_df['first_hit_distance'].append(initdist)
                    # time before first feeding
                    per_fly_df['first_hit_time'].append(np.nansum(dt[0:frame_firstarr]))
                    # time and distance on first trip
                    whichIndx = np.where(state==1)[0][0]+1
                    if whichIndx < len(pos):
                        frame_firstleave = pos[whichIndx] 
                        if num_visits > 1:
                            frame_2ndarr = pos[state==1][1]
                            per_fly_df['first_loop_time'].append(np.nansum(dt[frame_firstleave:frame_2ndarr]))
                            xVec = flyDf.body_x.values[frame_firstleave:frame_2ndarr]
                            yVec = flyDf.body_y.values[frame_firstleave:frame_2ndarr]
                            tripdist = np.nansum(np.sqrt((xVec[1:]-xVec[0:-1])**2 + (yVec[1:]-yVec[0:-1])**2))
                            per_fly_df['first_loop_distance'].append(tripdist)
                        else:
                            per_fly_df['first_loop_time'].append(np.nansum(dt[frame_firstleave:]))
                            xVec = flyDf.body_x.values[frame_firstleave:]
                            yVec = flyDf.body_y.values[frame_firstleave:]
                            tripdist = np.nansum(np.sqrt((xVec[1:]-xVec[0:-1])**2 + (yVec[1:]-yVec[0:-1])**2))
                            per_fly_df['first_loop_distance'].append(tripdist)
                        # CFF at the start of first trip
                        per_fly_df['CFF_afterfirstvisit'].append(CFF[frame_firstleave])
                    else:
                        per_fly_df['first_loop_time'].append(np.nan)
                        per_fly_df['first_loop_distance'].append(np.nan)
                        per_fly_df['CFF_afterfirstvisit'].append(np.nan)
                else:
                    per_fly_df['total_feeding'].append(np.nan)
                    per_fly_df['mean_visitdur'].append(np.nan)
                    per_fly_df['first_visitdur'].append(np.nan)
                    per_fly_df['first_visitdur_scaled'].append(np.nan)
                    per_fly_df['sumfirst2visitdur'].append(np.nan)
                    per_fly_df['mean_visitdur_after1'].append(np.nan)
                    per_fly_df['mean_visitdur_after2'].append(np.nan)
                    per_fly_df['first_hit_time'].append(np.nan)
                    per_fly_df['first_hit_distance'].append(np.nan)
                    per_fly_df['first_loop_time'].append(np.nan)
                    per_fly_df['first_loop_distance'].append(np.nan)
                    per_fly_df['CFF_afterfirstvisit'].append(np.nan)
                
    per_fly_df = pd.DataFrame(per_fly_df)
    return per_fly_df
    
def CreatePerTripDF(df,metadata):
    
    per_trip_df = {'fly': [], 
              'condition': [],
              'which_trip': [], # trip index (since encountering patch)
              'time': [], # time at the start of trip
              'CF': [], # cumulative feeding at start of trip
              'CFF': [],  # cumulative feeding fraction at start of trip (fraction of time feeding since start of expt)
              'trip_duration': [], # duration of trip
              'trip_distance': [], # distance travelled during trip  
              'trip_CFFchange': [], # change in CFF during trip
              'trip_relCFFchange': [], # change in CFF during trip  (relative to CFF at start of trip)
              'max_disp': [], # maximum displacement from center of food patch  
              'max_disp_prevtrips': [],
              'max_disp_sinceboundary': [], # maximum displacement from center of food patch across previous trips (since last encounter with arena boundary)
              'numtrips_sinceboundary': [], # number of trips made previously since last encounter with arena boundary 
              'priorVisit_duration': [], # duration of previous visit
              'priorVisit_CFFstart': [], # CFF at the start of previous visit 
              'priorVisit_CFFchange': [], # increase in CFF during previous visit 
              'priorVisit_relCFFchange': [], # increase in CFF (relative to max possible increase) during previous visit 
              'priorVisit_CFchange': [], # increase in CF during previous visit 
              'ifloop': [], # if this is a complete loop without hitting boundary
              'dist2dispRatio':[], # trip distance over maximum displacement 
              'dist2durationRatio':[], # trip distance over trip duration (average speed during trip)
            }

    for condition in tqdm(df.condition.unique()):
        for fly in tqdm(df.fly.unique()):
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]
            
            # location and properties of food spot as well as arena size
            food_x = metadata[fly]['arena']['spots']['x']
            food_y = metadata[fly]['arena']['spots']['y']
            
            # time between frames
            dt = flyDf.dt.values
            
            # displacement from center of food patch or arena
            # disp_center_all = flyDf.dist_center.values
            
            # duration of segment states
            segs_all = flyDf.segment.values
            runlen, pos, state, dur = rle(segs_all, dt=dt) # see function def above
            
            # frames just before animal leaves food source
            frameInds_beforeleaving = np.where(np.array((segs_all[:-1] == 1) & (segs_all[1:] != segs_all[:-1])))[0] 
            # frames when animal arrives at food source
            frameInds_arrival = np.where(np.array((segs_all[1:] == 1) & (segs_all[1:] != segs_all[:-1])))[0]+1
            
            # calculate number of trips
            if np.sum(state==1)>0:
                if segs_all[0] == 1: # if animal started off being on food spot
                    frameInds_arrival = np.append([0],frameInds_arrival)
                if segs_all[-1] == 1: # if animal ended on food spot
                    frameInds_beforeleaving = np.append(frameInds_beforeleaving,len(segs_all)-1)
                maxnumtrips = len(frameInds_beforeleaving)
                # visits = np.where(state==1)
                # visits_loop = np.where((state[:-1]==1) & (state[1:]==2))
            else:
                maxnumtrips = 0
            
            if maxnumtrips > 0:
                # max displacement when fly is feeding
                # maxdisp_feed = np.max(disp_center_all[flyDf.ethogram.values == 3])
    
                # max displacement when fly is considered to be on food patch
    #             maxdisp_food = np.max(disp_center_all[flyDf.segment.values == 1])
    
                tripIndx = 1
                maxtripdisp_sincestart = 0
                maxtripdisp_sincebound = 0 # maximum displacement of previous trips (since last boundary hit)
                numtrips_sincebound = 0
                for kk in range(maxnumtrips):
                    prevframe = frameInds_arrival[kk] # frame at which fly arrived at patch for the most recent visit (before leaving)
                    firstframe = frameInds_beforeleaving[kk]
                    if kk < maxnumtrips - 1:
                        endframe = frameInds_arrival[kk+1]
                    else:
                        endframe = len(segs_all)-1
                    xpos_all = flyDf.body_x.values[firstframe:endframe+1]
                    ypos_all = flyDf.body_y.values[firstframe:endframe+1]
                    totdist = np.nansum(np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2))
                    dist2patch = np.sqrt((xpos_all[1:-1]-food_x)**2 + (ypos_all[1:-1]-food_y)**2)
                    if ~np.isnan(np.array([dist2patch])).all():
#                     if ~np.isnan(np.array([np.nan,np.nan])).all():
                        maxdisp = np.nanmax(dist2patch)
                    else:
                        maxdisp = np.nan
    #                 if maxdisp > maxdisp_food: # only count a trip if the maximum displacement exceeds that when fly is on spot
                    segs_trip = flyDf.segment.values[firstframe:endframe+1]
                    triploop = (np.max(segs_trip) == 2)
    
                    # storage for data frame
                    # fly id and condition
                    per_trip_df['fly'].append(fly)
                    per_trip_df['condition'].append(condition)
    
                    # trip indxx
                    per_trip_df['which_trip'].append(tripIndx)
    
                    # time when trip first starts
                    per_trip_df['time'].append(flyDf.time.values[firstframe])
    
                    # CF when trip first starts
                    per_trip_df['CF'].append(flyDf.CF.values[firstframe])
                    
                    # CFF when trip first starts
                    per_trip_df['CFF'].append(flyDf.CFF.values[firstframe])
    
                    # duration of trip
                    duration = flyDf.time.values[endframe]-flyDf.time.values[firstframe]
                    per_trip_df['trip_duration'].append(duration)
    
                    # total distance travelled during trip
                    per_trip_df['trip_distance'].append(totdist)
                    
                    # CFF change during trip (absolute value)
                    CFFchange_trip = flyDf.CFF.values[firstframe]-flyDf.CFF.values[endframe-1]
                    per_trip_df['trip_CFFchange'].append(CFFchange_trip)
                    
                    # CFF change during trip (relative to max possible value)
                    relCFFchange_trip = CFFchange_trip/flyDf.CFF.values[firstframe]
                    per_trip_df['trip_relCFFchange'].append(relCFFchange_trip)
    
                    # maximum displacement from center of arena/food patch
                    per_trip_df['max_disp'].append(maxdisp)
    
                    # maximum displacement from center of arena/food patch across all previous trips
                    per_trip_df['max_disp_prevtrips'].append(maxtripdisp_sincestart)
    
                    # maximum displacement from center of food patch across previous trips (since last encounter with arena boundary)
                    per_trip_df['max_disp_sinceboundary'].append(maxtripdisp_sincebound)
    
                    # number of trips made previously since last encounter with arena boundary  
                    per_trip_df['numtrips_sinceboundary'].append(numtrips_sincebound)
    
                    # duration of prior visit (before leaving)
                    if prevframe > 0:
                        tprev = flyDf.time.values[prevframe-1]
                        CFprev = flyDf.CF.values[prevframe-1]
                        CFFprev = flyDf.CFF.values[prevframe-1]
                    else:
                        tprev = 0
                        CFprev = 0
                        CFFprev = 0
                    per_trip_df['priorVisit_duration'].append(flyDf.time.values[firstframe]-tprev)
                    # CFF at the start of prior visit
                    per_trip_df['priorVisit_CFFstart'].append(CFFprev)
                    # change in CFF during prior visit 
                    CFFchange = flyDf.CFF.values[firstframe]-CFFprev
                    per_trip_df['priorVisit_CFFchange'].append(CFFchange)
                    # change in CFF (relative to max possible increase) during prior visit 
                    relCFFchange = CFFchange/(1-CFFprev)
                    per_trip_df['priorVisit_relCFFchange'].append(relCFFchange)
                    
                    # increase in CF during previous visit 
                    CFchange = flyDf.CF.values[firstframe]-CFprev
                    per_trip_df['priorVisit_CFchange'].append(CFchange)
                    
                    # if trip is a loop (without hitting boundary)
                    per_trip_df['ifloop'].append(triploop)
                    
                    # distance/displacement ratio
                    per_trip_df['dist2dispRatio'].append(totdist/maxdisp)
                    
                    # distance/duration ratio
                    if duration > 0:
                        per_trip_df['dist2durationRatio'].append(totdist/duration)
                    else:
                        per_trip_df['dist2durationRatio'].append(np.nan)
    
                    # update indices and quantities
                    tripIndx = tripIndx + 1
                    maxtripdisp_sincestart = np.maximum(maxtripdisp_sincestart,maxdisp)
                    if triploop == True:
                        maxtripdisp_sincebound = np.maximum(maxtripdisp_sincebound,maxdisp)
                        numtrips_sincebound = numtrips_sincebound + 1
                    else:
                        maxtripdisp_sincebound = 0
                        numtrips_sincebound = 0
    
                            
    per_trip_df = pd.DataFrame(per_trip_df)
    per_trip_df = per_trip_df.loc[per_trip_df.trip_duration>0]
    
    # Augment data frame with next trip and visit properties 
    nextTrip_duration = [] # duration of next trip (after next visit)
    nextVisit_duration = [] # duration of next visit (after trip)
    nextVisit_CFFchange = [] # increase in CFF during next visit 
    nextVisit_relCFFchange = [] # increase in CFF (relative to max possible increase) during next visit 
                  
    for fly in tqdm(per_trip_df.fly.unique()):
        tripdur_rel = per_trip_df[per_trip_df.fly==fly].trip_duration.values
        nextTrip_duration.append(tripdur_rel[1:])
        nextTrip_duration.append(np.nan)
        
        priorVisitdur_rel = per_trip_df[per_trip_df.fly==fly].priorVisit_duration.values
        nextVisit_duration.append(priorVisitdur_rel[1:])
        nextVisit_duration.append(np.nan)
        
        priorVisitCFFchange_rel = per_trip_df[per_trip_df.fly==fly].priorVisit_CFFchange.values
        nextVisit_CFFchange.append(priorVisitCFFchange_rel[1:])
        nextVisit_CFFchange.append(np.nan)
        
        priorVisitrelCFFchange_rel = per_trip_df[per_trip_df.fly==fly].priorVisit_relCFFchange.values
        nextVisit_relCFFchange.append(priorVisitrelCFFchange_rel[1:])
        nextVisit_relCFFchange.append(np.nan)
        
        
    per_trip_df['nextTrip_duration'] = np.hstack(nextTrip_duration)
    per_trip_df['nextVisit_duration'] = np.hstack(nextVisit_duration)
    per_trip_df['nextVisit_CFFchange'] = np.hstack(nextVisit_CFFchange)
    per_trip_df['nextVisit_relCFFchange'] = np.hstack(nextVisit_relCFFchange)

    return per_trip_df

# Augment per-segment dataframe with previous segment properties
# Here we assume that the dataframe only consists of segments on trips
def AugmentSegDF_prevSegProp(perSegDF,qOIs):
    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict[qName + '_prev'] = []

    for fly in tqdm(perSegDF.fly.unique()):

        perfly_perSeg_DF = perSegDF[perSegDF.fly == fly]
        numtrips = int(np.max(perfly_perSeg_DF['after_which_visit'].values))

        for tripkk in range(numtrips):
            tripIndx = tripkk + 1
            relDF = perfly_perSeg_DF[perfly_perSeg_DF['after_which_visit'].values==tripIndx]
            if len(relDF) > 0:
                for qIndx in range(numqOIs):
                    qName = qOIs[qIndx]
                    qdata_rel = relDF[qName].values
                    if len(qdata_rel) > 0:
                        qdata_shifted = np.append(np.nan,qdata_rel[0:-1])
                        storagedict[qName + '_prev'].append(qdata_shifted)


    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        perSegDF[qName + '_prev'] = np.hstack(storagedict[qName + '_prev'])
    
    return perSegDF
    

# Augment segment dataframe with changes in successive segments
# Here we assume that the dataframe only consists of segments on trips
def AugmentSegDF_runturnchanges(perSegDF,qOIs):

    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict["d" + qName] = []
        
    for fly in tqdm(perSegDF.fly.unique()):

        perfly_perSeg_DF = perSegDF[perSegDF.fly == fly]
        numtrips = int(np.max(perfly_perSeg_DF['after_which_visit'].values))

        for tripkk in range(numtrips):
            tripIndx = tripkk + 1
            relDF = perfly_perSeg_DF[perfly_perSeg_DF['after_which_visit'].values==tripIndx]
            if len(relDF) > 0:
                for qIndx in range(numqOIs):
                    qName = qOIs[qIndx]
                    qdata_rel = relDF[qName].values
                    if len(qdata_rel) > 0:
                        dqdataVec = np.diff(qdata_rel)
                        dqdataVec = np.insert(dqdataVec,0,0)
                        storagedict["d" + qName].append(dqdataVec)
                
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        perSegDF["d" + qName] = np.hstack(storagedict["d" + qName])
    
    return perSegDF

# Augment seg DF with trip properties:
def AugmentSegDF_tripProps(perSegDf, per_trip_df, qOIs, nanReplacements):
    
    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict[qName] = []
    
    
    for fly in tqdm(perSegDf.fly.unique()):
        
        numvisits_fly = int(np.amax(perSegDf[perSegDf.fly==fly].after_which_visit.values))
        
        for visitIndx in range(numvisits_fly+1):
            perSegDf_aftervisit = perSegDf.loc[(perSegDf.fly==fly) & (perSegDf.after_which_visit==visitIndx)]
            numSegs = len(perSegDf_aftervisit)
            perTripDf_rel = per_trip_df.loc[(per_trip_df.fly==fly) & (per_trip_df.which_trip==visitIndx)]

            for qIndx in range(numqOIs):
                qName = qOIs[qIndx]
                if len(perTripDf_rel) > 0:                
                    q_rel = perTripDf_rel[qName].values[0]
                else:
                    q_rel = nanReplacements[qIndx] 
                
                storagedict[qName].append(q_rel*np.ones(numSegs))

    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        perSegDf[qName] = np.hstack(storagedict[qName])
    
    return perSegDf

# Augment seg DF with fly properties:
def AugmentSegDF_flyProps(perSegDf, per_fly_df, qOIs, nanReplacements):
    
    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict[qName] = []
    
    
    for fly in tqdm(perSegDf.fly.unique()):
        
        relFlyDF = per_fly_df[per_fly_df.fly == fly]
        perSegDf_perFly = perSegDf[perSegDf.fly==fly]
        numSegs_fly = len(perSegDf_perFly)
        
        for qIndx in range(numqOIs):
            qName = qOIs[qIndx]
            if numSegs_fly > 0:  
                q_rel = relFlyDF[qName].values                
            else:
                q_rel = nanReplacements[qIndx] 
            
            storagedict[qName].append(q_rel*np.ones(numSegs_fly))

    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        perSegDf[qName] = np.hstack(storagedict[qName])
    
    return perSegDf

# Augment Trip DF with segment properties:
# - number of movement segments during trip
# - turn bias during trip (max(#CW turns,#antiCW turns)/total#segs)
# - CWfrac during trip (#CWturns/total#segs)
# - meanabsheadturnangle: mean absolute head turn angle
# - meanlog10effArcRadius: mean effective arc radius
# - TurnAngle_bias: whether there is a difference in turn angle between preferred and non-preferred directions
# - effArcRadius_bias: whether there is a difference in curvature between preferred and non-preferred directions

# For some of the variables, we also have versions where we take into account only the food->food or food->border sections of the trip
def AugmentTripDF_SegProps(per_trip_df, perSegDf):
    
    numSegs = []
    turnbias = []
    CWfrac = []
    meanabsheadturnangle = []
    meanlog10effArcRadius = []
    headturnangle_bias = []
    log10effArcRadius_bias = []

    numSegs_toFoodOrBorder = []
    distance_toFoodOrBorder = []
    
    for fly in tqdm(per_trip_df.fly.unique()):

        numtrips_fly = np.amax(per_trip_df[per_trip_df.fly==fly].which_trip.values)
        
        for tripIndx in range(numtrips_fly):
            perSegDf_rel = perSegDf.loc[(perSegDf.fly==fly) & (perSegDf.after_which_visit==tripIndx+1) &
                                       (perSegDf.seg_state>=2) & (perSegDf.seg_state<=5)]
            perSegDf_toFoodOrBorder = perSegDf_rel[np.isin(perSegDf_rel.seg_state.values,[2,4])]
            
            absheadturnangle_rel = perSegDf_toFoodOrBorder.absheadturnangle.values
            effArcRadius_rel = perSegDf_toFoodOrBorder.effArcRadius.values
            ifCW_rel = perSegDf_toFoodOrBorder.ifCW.values
            ifpreferredturn_rel = perSegDf_toFoodOrBorder.ifpreferredturn.values
            segLengths_rel = perSegDf_toFoodOrBorder.seg_length.values
            # absheadturnangle_rel = perSegDf_rel.absheadturnangle.values
            # effArcRadius_rel = perSegDf_rel.effArcRadius.values
            # ifCW_rel = perSegDf_rel.ifCW.values
            # ifpreferredturn_rel = perSegDf_rel.ifpreferredturn.values
                        
            # number of runs on this trip
            totnumsegs = len(perSegDf_rel)
            numSegs.append(totnumsegs)

            # number of runs to food or border on this trip
            totnumsegs_toFoodOrBorder = len(perSegDf_toFoodOrBorder)
            numSegs_toFoodOrBorder.append(totnumsegs_toFoodOrBorder)

            if totnumsegs > 0:
                
                # number of CW turns
                numCWturns = np.sum(ifCW_rel)
                CWfrac_trip = numCWturns/totnumsegs_toFoodOrBorder
                # CWfrac_trip = numCWturns/totnumsegs
                turnbias_trip = 2*np.maximum(CWfrac_trip,1.0-CWfrac_trip)-1
                
                # mean turn angle and log10(curvature) among turns in preferred VS non-preferred direction
                meanabsheadturnangle_trip = np.nanmean(absheadturnangle_rel)
                log10effArcRadius_bias_trip = np.nanmean(np.log10(effArcRadius_rel))
                if ifpreferredturn_rel[0] != 0.5:
                    turnangles_preferred = absheadturnangle_rel[ifpreferredturn_rel == 1]
                    turnangles_nonpreferred = absheadturnangle_rel[ifpreferredturn_rel == 0]
                    log10effArcRadius_preferred = np.log10(effArcRadius_rel[ifpreferredturn_rel == 1])
                    log10effArcRadius_nonpreferred = np.log10(effArcRadius_rel[ifpreferredturn_rel == 0])
                    headturnangle_bias_rel = (np.nanmean(turnangles_preferred)-np.nanmean(turnangles_nonpreferred))/meanabsheadturnangle_trip
                    log10effArcRadius_bias_rel = ((np.nanmean(log10effArcRadius_preferred)-np.nanmean(log10effArcRadius_nonpreferred))
                                                  /log10effArcRadius_bias_trip)
                else:
                    headturnangle_bias_rel = np.nan
                    log10effArcRadius_bias_rel = np.nan
                    
                    
                # append desired quantities
                meanabsheadturnangle.append(meanabsheadturnangle_trip)
                meanlog10effArcRadius.append(log10effArcRadius_bias_trip)
                CWfrac.append(CWfrac_trip)
                turnbias.append(turnbias_trip)
                headturnangle_bias.append(headturnangle_bias_rel)
                log10effArcRadius_bias.append(log10effArcRadius_bias_rel)

                distance_toFoodOrBorder.append(np.nansum(segLengths_rel))
            
            else:
                meanabsheadturnangle.append(np.nan)
                meanlog10effArcRadius.append(np.nan)
                CWfrac.append(np.nan)
                turnbias.append(np.nan)
                headturnangle_bias.append(np.nan)
                log10effArcRadius_bias.append(np.nan)
                distance_toFoodOrBorder.append(np.nan)

    per_trip_df['numSegs'] = np.hstack(numSegs)
    per_trip_df['turnbias'] = np.hstack(turnbias)
    per_trip_df['CWfrac'] = np.hstack(CWfrac)
    per_trip_df['meanabsheadturnangle'] = np.hstack(meanabsheadturnangle)
    per_trip_df['meanlog10effArcRadius'] = np.hstack(meanlog10effArcRadius)
    per_trip_df['headturnangle_bias'] = np.hstack(headturnangle_bias)
    per_trip_df['log10effArcRadius_bias'] = np.hstack(log10effArcRadius_bias)
    per_trip_df['numSegs_toFoodOrBorder'] = np.hstack(numSegs_toFoodOrBorder)
    per_trip_df['distance_toFoodOrBorder'] = np.hstack(distance_toFoodOrBorder)
    
    return per_trip_df

# -------------------------------------------------------------------------------

# Augment Trip DF with properties scaled by average across all trips for each fly:
# - scaledTripDuration: tripduration/meanTripDurationOfFly
# - scaledTripDistance: tripdistance/meanTripDistanceOfFly
# - scaledTripDuration2: tripduration/meanLoopDurationOfFly
# - scaledTripDistance2: tripdistance/meanLoopDistanceOfFly
# - scaledPriorVisitDuration: priorVisitDuration/meanVisitDurationOfFly
# - scaledNextVisitDuration: nextVisitDuration/meanVisitDurationOfFly
def AugmentTripDF_scaledProps(per_trip_df, perFlyDf):
    scaledTripDuration = []
    scaledTripDistance = []
    scaledTripDuration2 = []
    scaledTripDistance2 = []
    scaledPriorVisitDuration = []
    scaledNextVisitDuration = []
    
    for fly in tqdm(per_trip_df.fly.unique()):

        meanTripDuration = perFlyDf[perFlyDf.fly==fly].meanTripDuration.values[0]
        meanTripDistance = perFlyDf[perFlyDf.fly==fly].meanTripDistance.values[0]
        meanLoopDuration = perFlyDf[perFlyDf.fly==fly].meanLoopDuration.values[0]
        meanLoopDistance = perFlyDf[perFlyDf.fly==fly].meanLoopDistance.values[0]
        meanVisitDuration = perFlyDf[perFlyDf.fly==fly].mean_visitdur.values[0]
        
        perTripDf_rel = per_trip_df.loc[per_trip_df.fly==fly]
        tripdur_rel = perTripDf_rel.trip_duration.values
        tripdist_rel = perTripDf_rel.trip_distance.values
        priorVisitdur_rel = perTripDf_rel.priorVisit_duration.values
        nextVisitdur_rel = perTripDf_rel.nextVisit_duration.values
                
        # append desired quantities
        scaledTripDuration.append(tripdur_rel/meanTripDuration)
        scaledTripDistance.append(tripdist_rel/meanTripDistance)
        scaledTripDuration2.append(tripdur_rel/meanLoopDuration)
        scaledTripDistance2.append(tripdist_rel/meanLoopDistance)
        scaledPriorVisitDuration.append(priorVisitdur_rel/meanVisitDuration)
        scaledNextVisitDuration.append(nextVisitdur_rel/meanVisitDuration)
                    
    per_trip_df['scaledTripDuration'] = np.hstack(scaledTripDuration)
    per_trip_df['scaledTripDistance'] = np.hstack(scaledTripDistance)
    per_trip_df['scaledTripDuration2'] = np.hstack(scaledTripDuration2)
    per_trip_df['scaledTripDistance2'] = np.hstack(scaledTripDistance2)
    per_trip_df['scaledPriorVisitDuration'] = np.hstack(scaledPriorVisitDuration)
    per_trip_df['scaledNextVisitDuration'] = np.hstack(scaledNextVisitDuration)
    
    return per_trip_df

# This is a function to augment tripDF with scaled trip properties without using per_fly_df
def AugmentTripDF_scaledProps_v2(per_trip_df, qOIs):

    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict['scaled_' + qName] = []
        storagedict['scaledbyloops_' + qName] = []
        
    for fly in tqdm(per_trip_df.fly.unique()):

        relDF = per_trip_df[per_trip_df.fly==fly]
        ifloop_rel = relDF.ifloop.values
        for qIndx in range(numqOIs):
            qName = qOIs[qIndx]
            qdata_rel = relDF[qName].values
            qmean = np.nanmean(qdata_rel)
            storagedict['scaled_' + qName].append(qdata_rel/qmean)
            if np.sum(ifloop_rel==True) > 0:
                qmeanLoop = np.nanmean(qdata_rel[ifloop_rel == True])
                storagedict['scaledbyloops_' + qName].append(qdata_rel/qmeanLoop)            
            else:
                storagedict['scaledbyloops_' + qName].append(np.nan*np.ones(len(relDF),))       
                
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        per_trip_df['scaled_' + qName] = np.hstack(storagedict['scaled_' + qName])
        per_trip_df['scaledbyloops_' + qName] = np.hstack(storagedict['scaledbyloops_' + qName])
    
    return per_trip_df



# Inputs:
# - thresvar: variable used to determine whether a trip is a long trip (or just considered small local search)
# - longthres: 
# Output:
# additional columns:
# - 'triptype': 0 = very short, local trip; 1 = long trip
# - 'boutIndx': which bout it is (starting from 0)
# - 'ifreset': 0 or 1 indicating whether this is a start of a bout of trips (if this is the first short trip after a long trip)
# = 'ifstartrisk': 0 or 1 indicating whether this is the first long trip (after one or more short trips)
# - 'maxtripdur_prevbout': maximum trip duration acrosss trips in previous bout
# - 'totaltripdur_prevbout': total trip duration acrosss trips in previous bout
# - 'CF_prevbout': cumulative feeding during previous bout
# - 'CF_currbout': cumulative feeding during current bout (at start of trip)
# - 'totaltripdur_currbout': total trip duration so far in current bout (including current trip)
def AugmentTripDF_triptype(per_trip_df, thresvar, longthres, outlierdetection = None):
    
    # varVec = per_trip_df[thresvar].values
    # triptype = (varVec > longthres)*1
    # per_trip_df['triptype'] = triptype 
    
    triptype = []
    boutIndx = []
    ifreset = [] 
    ifstartrisk = []
    maxtripduration_prevbout = []
    totaltripduration_prevbout = []
    CF_prevbout = []
    CF_currbout = []
    totaltripduration_currbout = []
    
    for fly in tqdm(per_trip_df.fly.unique()):
        # Determine trip type, whether it's the start of a new bout, or whether 
        # it's a transition from short trips to risky behavior
        varVec = per_trip_df[per_trip_df.fly==fly][thresvar].values
        numtrips = len(varVec)
        if outlierdetection == 'dynamic_medcouple':
            triptype_rel = np.zeros(numtrips)
            ifnewboutVec = np.zeros(numtrips)
            ifstartriskVec = np.zeros(numtrips)
            currbout_startIndx = 0
            triptype_prev = 0
            minthres = longthres
            shorttrip_ub = 0
            for tripIndx in range(numtrips):
                var_curr = varVec[tripIndx]
                if triptype_prev == 0:
                    if (tripIndx - currbout_startIndx) > 4:
                        varVec_rel = varVec[currbout_startIndx:tripIndx]
                    else:
                        varVec_rel = varVec[0:tripIndx]
                    if len(varVec_rel) > 4:
                        med = medcouple(varVec_rel)
                        Q3, Q1 = np.percentile(varVec_rel, [75 ,25])
                        IQR = Q3 - Q1
                        thres_curr = Q3 + np.exp(3*med)*1.5*IQR 
                        minthres = np.minimum(minthres,thres_curr)
                        
                triptype_curr = (var_curr > np.maximum(shorttrip_ub,minthres))*1
                triptype_rel[tripIndx] = triptype_curr
                
                if triptype_curr == 0:
                    shorttrip_ub = np.maximum(shorttrip_ub,var_curr)
                
                # determine if this is a new bout or new risky behavior
                if tripIndx > 0:
                    if triptype_curr > triptype_prev:
                        ifstartriskVec[tripIndx] = 1
                    elif triptype_curr < triptype_prev:
                        ifnewboutVec[tripIndx] = 1
                        currbout_startIndx = tripIndx
                        # minthres = longthres
                        shorttrip_ub = var_curr
                # update previous trip properties
                triptype_prev = triptype_curr
        else:
            if outlierdetection == None:
                thres = longthres
            elif outlierdetection == 'meanLowerHalf':
                thres = 2*np.mean(np.sort(varVec)[0:np.floor(numtrips/2)])
            elif outlierdetection == 'medcouple':
                if numtrips > 1:
                    M_fly = medcouple(varVec)
                    Q3, Q1 = np.percentile(varVec, [75 ,25])
                    IQR = Q3 - Q1
                    thres = Q3 + np.exp(3*M_fly)*1.5*IQR 
                else:
                    thres = np.max(varVec)
            triptype_rel = (varVec > thres)*1
            ifnewboutVec = np.diff(triptype_rel)==-1
            ifnewboutVec = np.insert(ifnewboutVec,0,0)
            ifstartriskVec = np.diff(triptype_rel)==1
            ifstartriskVec = np.insert(ifstartriskVec,0,0)
            
                    
        triptype.append(triptype_rel)
        ifreset.append(ifnewboutVec)
        ifstartrisk.append(ifstartriskVec)
        
        tripdur_rel = per_trip_df[per_trip_df.fly==fly].trip_duration.values
        priorVisit_CFchange_rel = per_trip_df[per_trip_df.fly==fly].priorVisit_CFchange.values
        
        maxtripdur_prev = np.nan
        totaltripdur_prev = np.nan
        CF_prev = np.nan
        maxtripdur_curr = 0
        totaltripdur_curr = 0
        CF_curr = 0
        boutIndx_curr = 0
        
        for tripIndx in range(numtrips):
            # is this trip the start of a new bout
            ifnewbout = ifnewboutVec[tripIndx]
            
            # if this is the start of a new bout, update previous and current bout trip durations
            if ifnewbout == 1:
                maxtripdur_prev = maxtripdur_curr
                totaltripdur_prev = totaltripdur_curr
                CF_prev = CF_curr
                maxtripdur_curr = 0
                totaltripdur_curr = 0
                CF_curr = 0
                boutIndx_curr = boutIndx_curr + 1
                
            # update CF right before trip
            CF_curr = CF_curr + priorVisit_CFchange_rel[tripIndx]
            
            # update total trip duration in current bout (including current trip)
            totaltripdur_curr = totaltripdur_curr + tripdur_rel[tripIndx]
            
            # update maximum trip duration in current bout
            maxtripdur_curr = np.maximum(maxtripdur_curr,tripdur_rel[tripIndx])
            
            # append storage arrays:
            maxtripduration_prevbout.append(maxtripdur_prev)
            totaltripduration_prevbout.append(totaltripdur_prev)
            CF_prevbout.append(CF_prev)
            totaltripduration_currbout.append(totaltripdur_curr)
            CF_currbout.append(CF_curr)
            boutIndx.append(boutIndx_curr)
                
        
    per_trip_df['triptype'] = np.hstack(triptype)
    per_trip_df['boutIndx'] = np.hstack(boutIndx)
    per_trip_df['ifreset'] = np.hstack(ifreset)
    per_trip_df['ifstartrisk'] = np.hstack(ifstartrisk)
    per_trip_df['maxtripduration_prevbout'] = np.hstack(maxtripduration_prevbout)
    per_trip_df['totaltripduration_prevbout'] = np.hstack(totaltripduration_prevbout)
    per_trip_df['CF_prevbout'] = np.hstack(CF_prevbout)
    per_trip_df['totaltripduration_currbout'] = np.hstack(totaltripduration_currbout)
    per_trip_df['CF_currbout'] = np.hstack(CF_currbout)
    
    
    return per_trip_df
    

# Augment dataframe with changes in successive trips
def AugmentTripDF_tripchanges(per_trip_df):
    
    dTripDuration = []
    dMaxDisp = []
    dTripDist = []
    dNextVisitDuration = []
    dPriorVisitDuration = []
    
    for fly in tqdm(per_trip_df.fly.unique()):

        tripdur_rel = per_trip_df[per_trip_df.fly==fly].trip_duration.values
        tripdist_rel = per_trip_df[per_trip_df.fly==fly].trip_distance.values
        maxdisp_rel = per_trip_df[per_trip_df.fly==fly].max_disp.values
        priorVisitdur_rel = per_trip_df[per_trip_df.fly==fly].priorVisit_duration.values
        nextVisitdur_rel = per_trip_df[per_trip_df.fly==fly].nextVisit_duration.values
        

        # how ariables has changed from previous trip
        dtripdurVec = np.diff(tripdur_rel)
        dtripdurVec = np.insert(dtripdurVec,0,0)
        dTripDuration.append(dtripdurVec)
              
        dtripdistVec = np.diff(tripdist_rel)
        dtripdistVec = np.insert(dtripdistVec,0,0)
        dTripDist.append(dtripdistVec)
        
        dmaxdispVec = np.diff(maxdisp_rel)
        dmaxdispVec = np.insert(dmaxdispVec,0,0)
        dMaxDisp.append(dmaxdispVec)
                
        dpriorVisitdurVec = np.diff(priorVisitdur_rel)
        dpriorVisitdurVec = np.insert(dpriorVisitdurVec,0,0)
        dPriorVisitDuration.append(dpriorVisitdurVec)
        
        dnextVisitdurVec = np.diff(nextVisitdur_rel)
        dnextVisitdurVec = np.insert(dnextVisitdurVec,0,0)
        dNextVisitDuration.append(dnextVisitdurVec)

    per_trip_df['dTripDuration'] = np.hstack(dTripDuration)
    per_trip_df['dMaxDisp'] = np.hstack(dMaxDisp)
    per_trip_df['dTripDist'] = np.hstack(dTripDist)
    per_trip_df['dNextVisitDuration'] = np.hstack(dNextVisitDuration)
    per_trip_df['dPriorVisitDuration'] = np.hstack(dPriorVisitDuration)
    
    
    return per_trip_df

# Create dataframe for the probability of returning to food patch without 
# hitting the arena boundary given that x > xthres for a fly, where x can be 
# variables such as maximum displaceemnt from food spot, trip duration, etc.
def CreatePreturnDF(per_trip_df, xVar, xthresVec):
    Preturn_df = {'fly': [], 
              'condition': [],
              'genotype': [], 
              'xthres': [], 
              'xGroup': [], 
              'Preturn': [], 
            }

    for condition in per_trip_df.condition.unique():
        for fly in per_trip_df.fly.unique():
            pertrip_flyDf = per_trip_df.loc[(per_trip_df.fly==fly)&(per_trip_df.condition==condition)]
            if len(pertrip_flyDf) > 0:
                gtype = pertrip_flyDf.genotype.values[0]
                xvals_all = pertrip_flyDf[xVar].values
                ifloop_all = pertrip_flyDf.ifloop.values
                
                for xIndx in range(len(xthresVec)):
                    xthres_val = xthresVec[xIndx]
                    ifloop_rel = ifloop_all[xvals_all > xthres_val]
                    if len(ifloop_rel) > 0:
                        Pr = np.sum(ifloop_rel)/len(ifloop_rel)
                    else:
                        Pr = np.nan
        
                    # append columns of dataframe
                    Preturn_df['fly'].append(fly)
                    Preturn_df['condition'].append(condition)
                    Preturn_df['genotype'].append(gtype)
                    Preturn_df['xthres'].append(xthres_val)
                    Preturn_df['xGroup'].append(str(xIndx))
                    Preturn_df['Preturn'].append(Pr)
                    
    Preturn_df = pd.DataFrame(Preturn_df)      
    
    return Preturn_df

# This is a function for augmenting the per-fly df with trip properties 
# (using the per-trip df)
def AugmentPerFlyDF_tripprops(per_fly_df,per_trip_df):
    
    numTrips = [] # number of trips after first encountering food spot
    numLoops = [] # number of trips that are loops
    Ploop = [] # probability that a trip is a loop
    meanTripDuration = [] # average trip duration 
    totexploretimeBefore2ndVisit = [] # total exploration time before 2nd visit    
    firstTripDuration = [] # first trip duration
    logfirstTripDuration = [] # log of first trip duration
    meanfirst3TripDurations = [] # average trip duration (among the first 3 trips)
    logmeanfirst3TripDurations = [] # log of average trip duration (among the first 3 trips)
    meanTripDistance = [] # average trip distance
    firstTripDistance = [] # first trip distance
    logfirstTripDistance = [] # log of first trip distance
    meanfirst3TripDistance = [] # average trip distance (among the first 3 trips)
    logmeanfirst3TripDistance = [] # log of average trip distance (among the first 3 trips)
    meanTripDisplacement = [] # average trip displacement
    firstTripDisplacement = [] # first trip displacement
    logfirstTripDisplacement = [] # log of first trip displacement
    meanfirst3TripDisplacement= [] # average trip displacement (among the first 3 trips)
    logmeanfirst3TripDisplacement = [] # log of average trip displacement (among the first 3 trips)
    meanLoopDuration = [] # average trip duration (among loops)
    meanLoopDistance = [] # average loop distance travelled during trip
    meanLoopDisplacement = [] # average loop max displacement from food spot
    
    for fly in tqdm(per_fly_df.fly.unique()):
        firsthittime = per_fly_df[per_fly_df.fly==fly].first_hit_time.values
        tripdur_rel = per_trip_df[per_trip_df.fly==fly].trip_duration.values
        tripdist_rel = per_trip_df[per_trip_df.fly==fly].trip_distance.values
        maxdisp_rel = per_trip_df[per_trip_df.fly==fly].max_disp.values
        ifloop_rel = per_trip_df[per_trip_df.fly==fly].ifloop.values
        numtrips = len(ifloop_rel)
        numTrips.append(numtrips)
        
        if numtrips > 0:
            numLoops.append(np.sum(ifloop_rel))
            Ploop.append(np.sum(ifloop_rel)/numtrips)
            meanTripDuration.append(np.nanmean(tripdur_rel))
            totexploretimeBefore2ndVisit.append(firsthittime + tripdur_rel[0])
            firstTripDuration.append(tripdur_rel[0])
            logfirstTripDuration.append(np.log10(tripdur_rel[0]))
            if np.sum(tripdur_rel[0:np.minimum(3,numtrips)]) > 0:
                meanfirst3TripDurations.append(np.nanmean(tripdur_rel[0:np.minimum(3,numtrips)]))
                logmeanfirst3TripDurations.append(np.log10(np.nanmean(tripdur_rel[0:np.minimum(3,numtrips)])))
            else:
                meanfirst3TripDurations.append(np.nan)
                logmeanfirst3TripDurations.append(np.nan)
            meanTripDistance.append(np.nanmean(tripdist_rel))
            firstTripDistance.append(tripdist_rel[0])
            if tripdist_rel[0] > 0:
                logfirstTripDistance.append(np.log10(tripdist_rel[0]))
            else:
                logfirstTripDistance.append(np.nan)
            if np.sum(tripdist_rel[0:np.minimum(3,numtrips)]) > 0:
                meanfirst3TripDistance.append(np.nanmean(tripdist_rel[0:np.minimum(3,numtrips)]))
                logmeanfirst3TripDistance.append(np.log10(np.nanmean(tripdist_rel[0:np.minimum(3,numtrips)])))
            else:
                meanfirst3TripDistance.append(np.nan)
                logmeanfirst3TripDistance.append(np.nan)
            meanTripDisplacement.append(np.nanmean(maxdisp_rel))
            firstTripDisplacement.append(maxdisp_rel[0])
            logfirstTripDisplacement.append(np.log10(maxdisp_rel[0]))
            meanfirst3TripDisplacement.append(np.nanmean(maxdisp_rel[0:np.minimum(3,numtrips)]))
            logmeanfirst3TripDisplacement.append(np.log10(np.nanmean(maxdisp_rel[0:np.minimum(3,numtrips)])))
            if np.sum(ifloop_rel==True) > 0:
                meanLoopDuration.append(np.nanmean(tripdur_rel[ifloop_rel==True]))
                meanLoopDistance.append(np.nanmean(tripdist_rel[ifloop_rel==True]))
                meanLoopDisplacement.append(np.nanmean(maxdisp_rel[ifloop_rel==True]))
            else:
                meanLoopDuration.append(np.nan)
                meanLoopDistance.append(np.nan)
                meanLoopDisplacement.append(np.nan)                
            
        else:
            numLoops.append(0)
            Ploop.append(np.nan)
            meanTripDuration.append(np.nan)
            totexploretimeBefore2ndVisit.append(firsthittime)
            firstTripDuration.append(np.nan)
            logfirstTripDuration.append(np.nan)
            meanfirst3TripDurations.append(np.nan)
            logmeanfirst3TripDurations.append(np.nan)
            meanTripDistance.append(np.nan)
            firstTripDistance.append(np.nan)
            logfirstTripDistance.append(np.nan)
            meanfirst3TripDistance.append(np.nan)
            logmeanfirst3TripDistance.append(np.nan)
            meanTripDisplacement.append(np.nan)
            firstTripDisplacement.append(np.nan)
            logfirstTripDisplacement.append(np.nan)
            meanfirst3TripDisplacement.append(np.nan)
            logmeanfirst3TripDisplacement.append(np.nan)
            meanLoopDuration.append(np.nan)
            meanLoopDistance.append(np.nan)
            meanLoopDisplacement.append(np.nan)
        
        
    per_fly_df['numTrips'] = np.hstack(numTrips)
    per_fly_df['numLoops'] = np.hstack(numLoops)
    per_fly_df['Ploop'] = np.hstack(Ploop)
    per_fly_df['meanTripDuration'] = np.hstack(meanTripDuration)
    per_fly_df['totexploretimeBefore2ndVisit'] = np.hstack(totexploretimeBefore2ndVisit)
    per_fly_df['firstTripDuration'] = np.hstack(firstTripDuration)
    per_fly_df['logfirstTripDuration'] = np.hstack(logfirstTripDuration)
    per_fly_df['meanfirst3TripDurations'] = np.hstack(meanfirst3TripDurations)
    per_fly_df['logmeanfirst3TripDurations'] = np.hstack(logmeanfirst3TripDurations)
    per_fly_df['meanTripDistance'] = np.hstack(meanTripDistance)
    per_fly_df['firstTripDistance'] = np.hstack(firstTripDistance)
    per_fly_df['logfirstTripDistance'] = np.hstack(logfirstTripDistance)
    per_fly_df['meanfirst3TripDistance'] = np.hstack(meanfirst3TripDistance)
    per_fly_df['logmeanfirst3TripDistance'] = np.hstack(logmeanfirst3TripDistance)
    per_fly_df['meanTripDisplacement'] = np.hstack(meanTripDisplacement)
    per_fly_df['firstTripDisplacement'] = np.hstack(firstTripDisplacement)
    per_fly_df['logfirstTripDisplacement'] = np.hstack(logfirstTripDisplacement)
    per_fly_df['meanfirst3TripDisplacement'] = np.hstack(meanfirst3TripDisplacement)
    per_fly_df['logmeanfirst3TripDisplacement'] = np.hstack(logmeanfirst3TripDisplacement)
    per_fly_df['meanLoopDuration'] = np.hstack(meanLoopDuration)
    per_fly_df['meanLoopDistance'] = np.hstack(meanLoopDistance)
    per_fly_df['meanLoopDisplacement'] = np.hstack(meanLoopDisplacement)
    
    return per_fly_df

# Function for augmenting per-frame dataframe with visit and trip indices, etc.
def AugmentPerFrameDF_tripVisitProps(df,per_trip_df,ifIncludeTriptype = False):
    # variables relating to which trip or visit:
    whichtrip = [] # if frame occurs during a trip, which trip is it 
    whichvisit = [] # if frame occurs during a visit, which visit is it
    currtripfrac_time = [] # if frame occurs during a trip, what fraction of the trip has passed (in terms of #frames/duration)
    currtripfrac_dist = [] # if frame occurs during a trip, what fraction of the trip has passed (in terms of distance)
    currvisitfrac = [] # if frame occurs during a visit, what fraction of the visit has passed (in terms of #frames/duration)
    ifloop = [] # if frame occurs during a loop trip (i.e. a trip that does not touch the wall)
    if ifIncludeTriptype == True:
        triptype = []
    
    for fly in tqdm(df.fly.unique()):
        time_frame = df[df.fly==fly].time.values
        time_starttrips = per_trip_df[per_trip_df.fly==fly].time.values
        tripduration = per_trip_df[per_trip_df.fly==fly].trip_duration.values
        time_endtrips = time_starttrips + tripduration
        priorVisitduration = per_trip_df[per_trip_df.fly==fly].priorVisit_duration.values
        time_startVisit = time_starttrips - priorVisitduration
        time_endVisit = time_starttrips
        
        # whether trips are loops
        ifloop_trips = per_trip_df[per_trip_df.fly==fly].ifloop.values
        if ifIncludeTriptype == True:
            triptype_trips = per_trip_df[per_trip_df.fly==fly].triptype.values
        
        # adjust start and end times by eps
        eps = 1e-5
        
        # remove last trip
        time_starttrips = time_starttrips[0:-1]
        time_endtrips = time_endtrips[0:-1]
        ifloop_trips = ifloop_trips[0:-1]
        if ifIncludeTriptype == True:
            triptype_trips = triptype_trips[0:-1]
            
        segs_all = df[df.fly==fly].segment.values
        if segs_all[-1] != 1: # tracking ends during a trip
            time_startVisit = time_startVisit[0:-1]
            time_endVisit = time_endVisit[0:-1]
            
        # distances
        bodyx_all = df[df.fly==fly].body_x.values
        bodyy_all = df[df.fly==fly].body_y.values
        
        whichtrip_fly = np.ones(len(time_frame))*np.nan
        currtripfrac_fly = np.ones(len(time_frame))*np.nan
        currtripfrac_dist_fly = np.ones(len(time_frame))*np.nan
        ifloop_fly = np.ones(len(time_frame))*np.nan
        if ifIncludeTriptype == True:
            triptype_fly = np.ones(len(time_frame))*np.nan
        numtrips = len(time_starttrips)
        for tripIndx in range(numtrips):
            starttime = time_starttrips[tripIndx]
            endtime = time_endtrips[tripIndx]
            startframe = np.where(time_frame > starttime + eps)[0][0]
            endframe = np.where(time_frame < endtime - eps)[0][-1]
            
            if endframe > startframe:
                whichtrip_fly[startframe:endframe+1] = tripIndx + 1
                currtripfrac_fly[startframe:endframe+1] = (np.arange(startframe,endframe+1)-startframe)/(endframe-startframe)
                
                bodyx_rel = bodyx_all[startframe:endframe+1]
                bodyy_rel = bodyy_all[startframe:endframe+1]
                dispVec = np.sqrt((bodyy_rel[1:] - bodyy_rel[0:-1])**2 + (bodyx_rel[1:] - bodyx_rel[0:-1])**2)
                if np.nansum(dispVec) > 0:
                    distVec = np.append(0,np.nancumsum(dispVec))/np.nansum(dispVec)
                else:
                    distVec = np.nan
                currtripfrac_dist_fly[startframe:endframe+1] = distVec
            else:
                whichtrip_fly[startframe] = tripIndx + 1
                currtripfrac_fly[startframe] = 0.5
                currtripfrac_dist_fly[startframe] = 0.5
            
            ifloop_currtrip = ifloop_trips[tripIndx]
            ifloop_fly[startframe:endframe+1] = ifloop_currtrip
            if ifIncludeTriptype == True:
                triptype_currtrip = triptype_trips[tripIndx]
                triptype_fly[startframe:endframe+1] = triptype_currtrip
            
            
        whichvisit_fly = np.ones(len(time_frame))*np.nan
        currvisitfrac_fly = np.ones(len(time_frame))*np.nan
        numvisits = len(time_startVisit)
        for visitIndx in range(numvisits):
            starttime = time_startVisit[visitIndx]
            endtime = time_endVisit[visitIndx]
            startframe = np.where(time_frame > starttime + eps)[0][0]
            endframe = np.where(time_frame < endtime + eps)[0][-1]
            if endframe > startframe:
                whichvisit_fly[startframe:endframe+1] = visitIndx + 1
                currvisitfrac_fly[startframe:endframe+1] = (np.arange(startframe,endframe+1)-startframe)/(endframe-startframe)
            else:
                whichvisit_fly[startframe] = visitIndx + 1
                currvisitfrac_fly[startframe] = 0.5
                
        # append to list
        whichtrip.append(whichtrip_fly)
        whichvisit.append(whichvisit_fly)
        currtripfrac_time.append(currtripfrac_fly)
        currtripfrac_dist.append(currtripfrac_dist_fly)
        currvisitfrac.append(currvisitfrac_fly)
        ifloop.append(ifloop_fly)
        if ifIncludeTriptype == True:
            triptype.append(triptype_fly)
            
    df['whichtrip'] = np.hstack(whichtrip)
    df['whichvisit'] = np.hstack(whichvisit)
    df['currtripfrac_time'] = np.hstack(currtripfrac_time)
    df['currtripfrac_dist'] = np.hstack(currtripfrac_dist)
    df['currvisitfrac'] = np.hstack(currvisitfrac)
    df['ifloop'] = np.hstack(ifloop)
    if ifIncludeTriptype == True:
        df['triptype'] = np.hstack(triptype)
    
    return df



# This is a function that creates a dataframe for wall to food segments 
def CreateWall2FoodDF(df):
    perWall2Food_df = {'fly': [], 
              'condition': [],
              'genotype': [], 
              'duration': [], 
              'distance': [], 
              'whichIndx': [], 
              'iffoodfound': [], 
            }

    for condition in df.condition.unique():
        for fly in df.fly.unique():
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]
            
            if len(flyDf) > 0:
            
                gtype = flyDf.genotype.values[0]
                
                # time between frames
                dt = flyDf.dt.values
                
                # duration of segment states
                segs_all = flyDf.segment.values
                runlen, pos, state, dur = rle(segs_all, dt=dt) 
                
                # # starting frames (start of wall to food segments)
                # startframeInds = np.where(np.array((segs_all[1:] == 3) & (segs_all[1:] != segs_all[:-1])))[0]+1
                # # ending frames (end of wall to food segments)
                # endframeInds = np.where(np.array((segs_all[:-1] == 3) & (segs_all[1:] != segs_all[:-1])))[0] 
                
                # indx of first visit
                visitInds = np.where(state == 1)[0]
                if len(visitInds) > 0:
                    firstVisit_indx = visitInds[0]
                    
                    wall2foodsegInds = np.where(state == 3)[0]
                    
                    for kk in range(len(wall2foodsegInds)):
                        segIndx = wall2foodsegInds[kk]
                        pos_seg = pos[segIndx]
                        dur_seg = dur[segIndx]
                        # startframe = startframeInds[kk]
                        # endframe = endframeInds[kk]
                        startframe = pos_seg
                        if segIndx < len(pos)-1:
                            endframe = pos[segIndx+1]-1
                        else:
                            endframe = len(segs_all)
                    
                        xpos_all = flyDf.body_x.values[startframe:endframe+1]
                        ypos_all = flyDf.body_y.values[startframe:endframe+1]
                        totdist = np.nansum(np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2))
                            
                    
                        # append columns of dataframe
                        perWall2Food_df['fly'].append(fly)
                        perWall2Food_df['condition'].append(condition)
                        perWall2Food_df['genotype'].append(gtype)
                        perWall2Food_df['duration'].append(dur_seg)
                        perWall2Food_df['distance'].append(totdist)
                        perWall2Food_df['whichIndx'].append(kk+1)
                        perWall2Food_df['iffoodfound'].append((pos_seg>firstVisit_indx))
                    
    perWall2Food_df = pd.DataFrame(perWall2Food_df)      
    
    return perWall2Food_df


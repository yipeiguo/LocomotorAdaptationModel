# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:07:47 2022

@author: labadmin
"""

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
def CreatePerMoveSegmentDF(df,metadata,ethogramOI):
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
                  'netReorientationAngle': [], # net change in angle in movement direction between the end and start of segment
                  'absReorientAngle': [], # absolute value of reorientation angle
                  'startmoveAngle_rel2food': [], # movement angle at start of segment (relative to the vector from food to current position)  
                }
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
                        headingAngle_all = flyDf.angle.values[startframe:endframe]
                        headturnangle_currseg = computeTurnAngBetween2angles(headingAngle_all[0],headingAngle_all[-1])
                        moveSegDf_fly['headturnangle'].append(headturnangle_currseg)
                        moveSegDf_fly['absheadturnangle'].append(np.abs(headturnangle_currseg))

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
                        
                if (segtype_curr == 4) or (segtype_curr == 5) or (segtype_curr == 0):
                    cumtime_curr = cumtime_curr + etho_StartTime[-1]
           
            moveSegDf_fly = pd.DataFrame(moveSegDf_fly)

            perMoveSegDf = perMoveSegDf.append(moveSegDf_fly, sort=False)
            
    return perMoveSegDf

# function for creating PerMoveSegment dataframe
# These movement segments can be runs or turns or both, etc. specified through 'ethogramOI'
# In this v2, the turn angles are between -2*pi and pi so that we can distinguish between small turns (< pi) and big turns (>pi).
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
                  'netReorientationAngle': [], # net change in angle in movement direction between the end and start of segment
                  'absReorientAngle': [], # absolute value of reorientation angle
                  'startmoveAngle_rel2food': [], # movement angle at start of segment (relative to the vector from food to current position)  
                }
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
                        else:
                            moveSegDf_fly['numDirChanges'].append(0)
                            moveSegDf_fly['ifbigturn'].append(0)
                            moveSegDf_fly['headturnangle'].append(0)
                            moveSegDf_fly['absheadturnangle'].append(0)

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
                        
                if (segtype_curr == 4) or (segtype_curr == 5) or (segtype_curr == 0):
                    cumtime_curr = cumtime_curr + etho_StartTime[-1]
           
            moveSegDf_fly = pd.DataFrame(moveSegDf_fly)

            perMoveSegDf = perMoveSegDf.append(moveSegDf_fly, sort=False)
            
    return perMoveSegDf

# function for creating PerRunSegment dataframe
def CreatePerRunDF(df,metadata):
    perRunDf = pd.DataFrame()
    for condition in tqdm(df.condition.unique()):
        for fly in tqdm(df.fly.unique()):
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]

            # location and properties of food spot as well as arena size
            food_x = metadata[fly]['arena']['spots']['x']
            food_y = metadata[fly]['arena']['spots']['y']

            segL, segSt, segType = rle(flyDf.segment)

            # runDf_fly = pd.DataFrame()
            runDf_fly = {'fly': [], 
                  'condition': [],
                  'seg_state': [],  
                  'after_which_visit': [],
                  'dist_since_visit': [], # total distance travelled during runs since last food spot visit
                  'time_since_visit': [], # time since last food spot visit
                  'cumRunTime_since_visit': [], # total run time since last food spot visit       
                  'which_run': [],
                  'run_duration': [],
                  'run_length': [],
                  'init_disp': [],
                  'run_disp': [],
                  'velo': [],  
                  'run_angle': [],       
                  'move_relangle': [],  
                  'turn_angle_fromprev': [],                     
                }
            prev_visit = 0
            cumdist_curr = 0
            cumRunTime_curr = 0                    
            cumtime_curr = 0
            runfound = False
            for ii, ss in enumerate(segSt):
                # runDf_seg = pd.DataFrame()
                segtype_curr = segType[ii]
                if segtype_curr == 1:
                    prev_visit = prev_visit + 1
                if (segtype_curr == 1) or (segtype_curr == 2) or (segtype_curr == 4):
                    cumdist_curr = 0
                    cumRunTime_curr = 0
                    cumtime_curr = 0 # reset time since leaving food spot

                se = min(ss+segL[ii], len(flyDf.body_x.values))
    #             posX = flyDf.body_x.values[ss:se]
    #             posY = flyDf.body_y.values[ss:se]
    #             cspath = np.cumsum( np.hypot( np.hstack((0, np.diff(posX) )),
    #                                  np.hstack((0, np.diff(posY) )) ) )
    #             velo = ( np.hypot( np.hstack((0, np.diff(posX) )),
    #                                  np.hstack((0, np.diff(posY) )) ) )/flyDf.dt[ss:se]

                ethoL, ethoSt, ethoType, ethoDuration = rle(flyDf.ethogram[ss:se], flyDf.dt[ss:se])
                runSegs = np.where(ethoType==2)[0]
                etho_StartTime = np.cumsum(np.insert(ethoDuration,0,0))
                runSegs_starttime = etho_StartTime[runSegs]

                if len(runSegs) > 0: 
                    for runIndx in range(len(runSegs)):

                        # storage for data frame
                        # fly id and condition
                        runDf_fly['fly'].append(fly)
                        runDf_fly['condition'].append(condition)

                        # segment state
                        runDf_fly['seg_state'].append(segtype_curr)

                        # number of food visits (entrance to food spot) before this run
                        runDf_fly['after_which_visit'].append(prev_visit)

                        # which run segment is it since entering this segment state
                        runDf_fly['which_run'].append(runIndx)

                        # run duration
                        rundur = ethoDuration[runSegs[runIndx]]
                        runDf_fly['run_duration'].append(rundur)
                        # store total running time since last visit
                        runDf_fly['cumRunTime_since_visit'].append(cumRunTime_curr)
                        if segtype_curr != 1:
                            cumRunTime_curr = cumRunTime_curr + rundur
                        # store total time since last visit
                        runDf_fly['time_since_visit'].append(cumtime_curr + runSegs_starttime[runIndx])
                        
                        
                        # run length (distance)
                        startframe = max(ss + ethoSt[runSegs[runIndx]],0)
                        endframe = min(ss + ethoSt[runSegs[runIndx]] + ethoL[runSegs[runIndx]] + 1, len(flyDf.body_x.values))
                        xpos_all = flyDf.body_x.values[startframe:endframe]
                        ypos_all = flyDf.body_y.values[startframe:endframe]
                        dist_all = np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2)
                        totdist = np.nansum(dist_all)
                        runDf_fly['run_length'].append(totdist)
                        # store total running distance since last visit
                        runDf_fly['dist_since_visit'].append(cumdist_curr)                        
                        if segtype_curr != 1:
                            cumdist_curr = cumdist_curr + totdist
                            
                        # starting displacement from center of spot
                        runDf_fly['init_disp'].append(np.sqrt((xpos_all[0]-food_x)**2 + (ypos_all[0]-food_y)**2))

                        # run length (displacement)
                        disp = np.sqrt((xpos_all[-1]-xpos_all[0])**2 + (ypos_all[-1]-ypos_all[0])**2)
                        runDf_fly['run_disp'].append(disp)

                        # mean velocity
                        dt_all = flyDf.dt.values[startframe:endframe-1]
                        vel_all = dist_all/dt_all
                        runDf_fly['velo'].append(np.mean(vel_all))

                        # run angle (wrt x-axis)
                        angle_curr = math.atan2(ypos_all[-1]-ypos_all[0],xpos_all[-1]-xpos_all[0])
                        runDf_fly['run_angle'].append(angle_curr)
                        if runfound == False:
                            angle_prev = angle_curr
                            runfound = True

                        # travelling angle (wrt food source)
                        relangle = (math.atan2(ypos_all[-1]-ypos_all[0],xpos_all[-1]-xpos_all[0]) - 
                            math.atan2(ypos_all[0]-food_y,xpos_all[0]-food_x))
                        if relangle > np.pi:
                            relangle = relangle - 2.*np.pi
                        elif relangle < -np.pi:
                            relangle = relangle + 2.*np.pi
                        runDf_fly['move_relangle'].append(relangle)

                        # turning angle (from previous angle)
                        turnangle = angle_curr - angle_prev
                        if turnangle > np.pi:
                            turnangle = turnangle - 2.*np.pi
                        elif turnangle < -np.pi:
                            turnangle = turnangle + 2.*np.pi
                        runDf_fly['turn_angle_fromprev'].append(turnangle)
                        angle_prev = angle_curr

                if (segtype_curr == 4) or (segtype_curr == 5) or (segtype_curr == 0):
                    cumtime_curr = cumtime_curr + etho_StartTime[-1]
                        


            runDf_fly = pd.DataFrame(runDf_fly)

            perRunDf = perRunDf.append(runDf_fly, sort=False)
            
    return perRunDf

# function for creating PerRunSegment dataframe
# In this v2, instead of extracting run segments and treating them as individual runs separated by turns,
# we extract the turn segments and assume that turns are separated by runs (i.e. we use the mid point of turn segments to separate out individual runs)
def CreatePerRunDF_v2(df,metadata):
    perRunDf = pd.DataFrame()
    for condition in tqdm(df.condition.unique()):
        for fly in tqdm(df.fly.unique()):
            flyDf = df.loc[(df.fly==fly)&(df.condition==condition)]

            # location and properties of food spot as well as arena size
            food_x = metadata[fly]['arena']['spots']['x']
            food_y = metadata[fly]['arena']['spots']['y']

            segL, segSt, segType = rle(flyDf.segment)

            # runDf_fly = pd.DataFrame()
            runDf_fly = {'fly': [], 
                  'condition': [],
                  'seg_state': [],  
                  'after_which_visit': [],
                  'dist_since_visit': [], # total distance travelled since last food spot visit
                  'time_since_visit': [], # time since last food spot visit
                  'which_run': [],
                  'run_duration': [],
                  'run_length': [],
                  'init_disp': [],
                  'run_disp': [],
                  'velo': [],  
                  'run_angle': [],       
                  'move_relangle': [],  
                  'turn_angle_fromprev': [],                     
                }
            prev_visit = 0
            cumdist_curr = 0
            cumtime_curr = 0
            runfound = False
            for ii, ss in enumerate(segSt):
                # runDf_seg = pd.DataFrame()
                segtype_curr = segType[ii]
                if segtype_curr == 1:
                    prev_visit = prev_visit + 1
                if (segtype_curr == 1) or (segtype_curr == 2) or (segtype_curr == 3):
                    cumdist_curr = 0
                    cumtime_curr = 0

                se = min(ss+segL[ii], len(flyDf.body_x.values))
    #             posX = flyDf.body_x.values[ss:se]
    #             posY = flyDf.body_y.values[ss:se]
    #             cspath = np.cumsum( np.hypot( np.hstack((0, np.diff(posX) )),
    #                                  np.hstack((0, np.diff(posY) )) ) )
    #             velo = ( np.hypot( np.hstack((0, np.diff(posX) )),
    #                                  np.hstack((0, np.diff(posY) )) ) )/flyDf.dt[ss:se]
                segDf = flyDf[ss:se]
                ethoL, ethoSt, ethoType, ethoDuration = rle(segDf.ethogram, segDf.dt)
                turnSegs = np.where(ethoType==1)[0] 
                # runSegs = np.where(ethoType==2)[0]
                numTurns = len(turnSegs)
                numRuns = numTurns + 1
                # Turn frame (assume it's the middle of turn segment)
                turnFrames = np.floor(ethoSt[turnSegs] + ethoL[turnSegs]/2).astype(int)
                
                # start frame for initial run in this segment
                startframe = 0
                runIndx = 0
                for potrunIndx in range(numRuns):
                    # print('startframe:',startframe)
                    # storage for data frame
                    
                    # end frame of run is turn frame
                    if potrunIndx < numRuns - 1:
                        endframe = turnFrames[potrunIndx]
                    else:
                        endframe = len(segDf)-1
                    # print('endframe:',endframe)
                    
                    # dataframe for current run segment 
                    currRunSegDF = segDf[startframe:endframe]
                    currRunSegDF = currRunSegDF[np.isin(currRunSegDF.ethogram,[2])]
                    # currRunSegDF = currRunSegDF[np.isin(currRunSegDF.ethogram,[1,2])]

                    if len(currRunSegDF) > 0:

                        # fly id and condition
                        runDf_fly['fly'].append(fly)
                        runDf_fly['condition'].append(condition)
    
                        # segment state
                        runDf_fly['seg_state'].append(segtype_curr)
    
                        # number of food visits (entrance to food spot) before this run
                        runDf_fly['after_which_visit'].append(prev_visit)
                        
                        # which run segment is it since entering this segment state
                        runDf_fly['which_run'].append(runIndx)
                        runIndx = runIndx + 1
    
                        # run duration
                        rundur = np.sum(currRunSegDF.dt.values)
                        runDf_fly['run_duration'].append(rundur)
                        # store total running time since last visit
                        runDf_fly['time_since_visit'].append(cumtime_curr)
                        if segtype_curr != 1:
                            cumtime_curr = cumtime_curr + rundur
                            
                        # run length (distance)
                        xpos_all = currRunSegDF.body_x.values
                        ypos_all = currRunSegDF.body_y.values
                        dist_all = np.sqrt((xpos_all[1:]-xpos_all[:-1])**2 + (ypos_all[1:]-ypos_all[:-1])**2)
                        totdist = np.nansum(dist_all)
                        runDf_fly['run_length'].append(totdist)
                        # store total running distance since last visit
                        runDf_fly['dist_since_visit'].append(cumdist_curr)                        
                        if segtype_curr != 1:
                            cumdist_curr = cumdist_curr + totdist
                        
    
                        # starting displacement from center of spot
                        runDf_fly['init_disp'].append(np.sqrt((xpos_all[0]-food_x)**2 + (ypos_all[0]-food_y)**2))
    
                        # run length (displacement)
                        disp = np.sqrt((xpos_all[-1]-xpos_all[0])**2 + (ypos_all[-1]-ypos_all[0])**2)
                        runDf_fly['run_disp'].append(disp)
    
                        # mean velocity
                        dt_all = currRunSegDF.dt.values[0:-1]
                        vel_all = dist_all/dt_all
                        runDf_fly['velo'].append(np.mean(vel_all))
    
                        # run angle (wrt x-axis)
                        angle_curr = math.atan2(ypos_all[-1]-ypos_all[0],xpos_all[-1]-xpos_all[0])
                        runDf_fly['run_angle'].append(angle_curr)
                        if runfound == False:
                            angle_prev = angle_curr
                            runfound = True
    
                        # travelling angle (wrt food source)
                        relangle = (math.atan2(ypos_all[-1]-ypos_all[0],xpos_all[-1]-xpos_all[0]) - 
                            math.atan2(ypos_all[0]-food_y,xpos_all[0]-food_x))
                        if relangle > np.pi:
                            relangle = relangle - 2.*np.pi
                        elif relangle < -np.pi:
                            relangle = relangle + 2.*np.pi
                        runDf_fly['move_relangle'].append(relangle)
    
                        # turning angle (from previous angle)
                        turnangle = angle_curr - angle_prev
                        if turnangle > np.pi:
                            turnangle = turnangle - 2.*np.pi
                        elif turnangle < -np.pi:
                            turnangle = turnangle + 2.*np.pi
                        runDf_fly['turn_angle_fromprev'].append(turnangle)
                        angle_prev = angle_curr

                        # update start frame of next run
                        startframe = endframe

            runDf_fly = pd.DataFrame(runDf_fly)

            perRunDf = perRunDf.append(runDf_fly, sort=False)
            # perRunDf = pd.concat(perRunDf,runDf_fly)
            
    return perRunDf


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

# Augment run DF with scaled variables:
def AugmentRunDF_scaledProp(perRunDf):
    scaledRL = [] # run length divided by the meanRL during that trip or visit
    scaledLog10RL = [] # log10(RL) - meanLog10(RL) during that trip or visit
    scaledRunDuration = [] # run duration divided by the mean run durations during that trip or visit
    absTurnAngle = [] # absolute turn angle
    scaledTurnAngle = [] # (absolute turn angle)/(mean absolute turn angle during trip or visit)
    scaledLog10TurnAngle = [] # log10(|turnangle|) - meanLog10(|turnangle|)
    ifCW = [] # if clockwise turn
    ifpreferredturn = [] # if turn is in preferred direction (defined as the direction that fly turns most of the time)
    
    # for extracting local rules:
    rundisp_prev = [] # run displacement of previous run
    runduration_prev = [] # run duration of previous run
    velo_prev = [] # velocity of previous run
    RL_prev = [] # runlength of previous run
    scaledRL_prev = [] # scaled runlength of the previous run
    scaledLog10RL_prev = [] # scaled log10(RL) of the previous run
    turnangle_prev = [] # turn angle of previous run (negative if CW)
    absTurnAngle_prev = [] # absolute turn angle of previous run
    scaledTurnAngle_prev = [] # scaled absolute turn angle of previous run
    scaledLog10TurnAngle_prev = [] 
    ifCW_prev = [] # if previous turn was clockwise
    ifpreferredturn_prev = [] 
    
    for fly in tqdm(perRunDf.fly.unique()):
        
        numvisits_fly = int(np.amax(perRunDf[perRunDf.fly==fly].after_which_visit.values))
#         numvisits_fly = int(perRunDf[perRunDf.fly==fly].after_which_visit.values[-1])
        
        for visitIndx in range(numvisits_fly+1):
            perRunDf_aftervisit = perRunDf.loc[(perRunDf.fly==fly) & (perRunDf.after_which_visit==visitIndx)]
            
            if len(perRunDf_aftervisit) > 0:
                # segment states after this visit
                segstates = perRunDf_aftervisit.seg_state.unique()

                if segstates[0] == 1: # if there are runs on the food spot visit
                    numRunTypes = 2
                else: 
                    numRunTypes = 1

                for runtypeIndx in range(numRunTypes):
                    if (runtypeIndx+1) == numRunTypes:
                        perRunDf_rel = perRunDf_aftervisit[perRunDf_aftervisit.seg_state!=1]
                    else:
                        perRunDf_rel = perRunDf_aftervisit[perRunDf_aftervisit.seg_state==1]
                    numRuns = len(perRunDf_rel)
                    if numRuns > 0:
                        rundisp_allrel = perRunDf_rel.run_disp.values
                        runlengths_allrel = perRunDf_rel.run_length.values
                        rundurations_allrel = perRunDf_rel.run_duration.values
                        velo_allrel = perRunDf_rel.velo.values
                        turnangles_allrel = perRunDf_rel.turn_angle_fromprev.values

                        log10RL_allrel = np.log10(runlengths_allrel)
                        absTurnAngles_allrel = np.absolute(turnangles_allrel)
                        log10TurnAngles_allrel = np.log10(absTurnAngles_allrel)
                        ifCW_allrel = (turnangles_allrel<0)
                        numCWturns = np.sum(ifCW_allrel)
                        if numCWturns > (numRuns/2.0):
                            ifpref_allrel = ifCW_allrel
                        else:
                            ifpref_allrel = ~ifCW_allrel


        #                 # remove non-relevant values (zeros and nans)
        #                 rundurations_rel = rundurations_allrel[runlengths_allrel>0]
        #                 runlengths_rel = runlengths_allrel[runlengths_allrel>0]
        #                 rundurations_rel = rundurations_rel[~np.isnan(runlengths_rel)]
        #                 runlengths_rel = runlengths_rel[~np.isnan(runlengths_rel)]

        #                 log10RL_rel = np.log10(runlengths_rel)

                        # means
                        meanRL = np.nanmean(runlengths_allrel)
                        meanlog10RL = np.nanmean(np.log10(runlengths_allrel[runlengths_allrel>0]))
                        meanRunDuration = np.nanmean(rundurations_allrel)
                        meanTurnAngle = np.nanmean(absTurnAngles_allrel)
                        meanlog10TurnAngle = np.nanmean(log10TurnAngles_allrel)

                        # scaled variables
                        scaledRL_rel = runlengths_allrel/meanRL
                        scaledLog10RL_rel = log10RL_allrel - meanlog10RL
                        if meanRunDuration != 0 :
                            scaledRunDuration_rel = rundurations_allrel/meanRunDuration
                        else:
                            scaledRunDuration_rel = np.empty(numRuns,)*np.nan
                        if meanTurnAngle != 0:
                            scaledTurnAngle_rel = absTurnAngles_allrel/meanTurnAngle
                        else:
                            scaledTurnAngle_rel = np.empty(numRuns,)*np.nan
                        scaledLog10TurnAngle_rel = log10TurnAngles_allrel - meanlog10TurnAngle

                        prevRunDisp_rel = np.append(np.nan,rundisp_allrel[0:-1])
                        prevRunDurations_rel = np.append(np.nan,rundurations_allrel[0:-1])
                        prevVelo_rel = np.append(np.nan,velo_allrel[0:-1])
                        prevRL_rel = np.append(np.nan,runlengths_allrel[0:-1])
                        prevscaledRL_rel = np.append(np.nan,scaledRL_rel[0:-1])
                        prevscaledLog10RL_rel = np.append(np.nan,scaledLog10RL_rel[0:-1])
                        prevturnangle_rel = np.append(np.nan,turnangles_allrel[0:-1])
                        prevabsturnangle_rel = np.append(np.nan,absTurnAngles_allrel[0:-1])
                        prevscaledTurnAngle_rel = np.append(np.nan,scaledTurnAngle_rel[0:-1])
                        prevscaledLog10TurnAngle_rel = np.append(np.nan,scaledLog10TurnAngle_rel[0:-1])

                        previfCW_rel = np.append(np.nan,ifCW_allrel[0:-1])
                        previfpref_rel = np.append(np.nan,ifpref_allrel[0:-1])

                        # append to storage arrays
                        scaledRL.append(scaledRL_rel)
                        scaledLog10RL.append(scaledLog10RL_rel)
                        scaledRunDuration.append(scaledRunDuration_rel)
                        absTurnAngle.append(absTurnAngles_allrel)
                        scaledTurnAngle.append(scaledTurnAngle_rel)
                        scaledLog10TurnAngle.append(scaledLog10TurnAngle_rel)
                        ifCW.append(ifCW_allrel)
                        ifpreferredturn.append(ifpref_allrel)

                        rundisp_prev.append(prevRunDisp_rel)
                        runduration_prev.append(prevRunDurations_rel)
                        velo_prev.append(prevVelo_rel)
                        RL_prev.append(prevRL_rel)
                        scaledRL_prev.append(prevscaledRL_rel)
                        scaledLog10RL_prev.append(prevscaledLog10RL_rel)
                        turnangle_prev.append(prevturnangle_rel)
                        absTurnAngle_prev.append(prevabsturnangle_rel)
                        scaledTurnAngle_prev.append(prevscaledTurnAngle_rel)
                        scaledLog10TurnAngle_prev.append(prevscaledLog10TurnAngle_rel)

                        ifCW_prev.append(previfCW_rel)
                        ifpreferredturn_prev.append(previfpref_rel)

                
                
    perRunDf['scaledRL'] = np.hstack(scaledRL)
    perRunDf['scaledLog10RL'] = np.hstack(scaledLog10RL)
    perRunDf['scaledRunDuration'] = np.hstack(scaledRunDuration)
    perRunDf['absTurnAngle'] = np.hstack(absTurnAngle)
    perRunDf['scaledTurnAngle'] = np.hstack(scaledTurnAngle)
    perRunDf['scaledLog10TurnAngle'] = np.hstack(scaledLog10TurnAngle)
    perRunDf['ifCW'] = np.hstack(ifCW)
    perRunDf['ifpreferredturn'] = np.hstack(ifpreferredturn)
    
    perRunDf['rundisp_prev'] = np.hstack(rundisp_prev)
    perRunDf['runduration_prev'] = np.hstack(runduration_prev)
    perRunDf['velo_prev'] = np.hstack(velo_prev)
    perRunDf['RL_prev'] = np.hstack(RL_prev)
    perRunDf['scaledRL_prev'] = np.hstack(scaledRL_prev)
    perRunDf['scaledLog10RL_prev'] = np.hstack(scaledLog10RL_prev)
    perRunDf['turnangle_prev'] = np.hstack(turnangle_prev)
    perRunDf['absTurnAngle_prev'] = np.hstack(absTurnAngle_prev)
    perRunDf['scaledTurnAngle_prev'] = np.hstack(scaledTurnAngle_prev)
    perRunDf['scaledLog10TurnAngle_prev'] = np.hstack(scaledLog10TurnAngle_prev)
    
    perRunDf['ifCW_prev'] = np.hstack(ifCW_prev)
    perRunDf['ifpreferredturn_prev'] = np.hstack(ifpreferredturn_prev)
    
    return perRunDf

# Augment run dataframe with changes in successive runs and turns
# Here we assume that the dataframe only consists of runs on trips
def AugmentRunDF_runturnchanges(perRunDF,qOIs):

    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict["d" + qName] = []
        
    for fly in tqdm(perRunDF.fly.unique()):

        perfly_perRun_DF = perRunDF[perRunDF.fly == fly]
        numtrips = int(np.max(perfly_perRun_DF['after_which_visit'].values))

        for tripkk in range(numtrips):
            tripIndx = tripkk + 1
            relDF = perfly_perRun_DF[perfly_perRun_DF['after_which_visit'].values==tripIndx]
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
        perRunDF["d" + qName] = np.hstack(storagedict["d" + qName])
    
    return perRunDF

# Augment run DF with trip properties:
def AugmentRunDF_tripProps(perRunDf, per_trip_df):
    triptypes = [] # whether run belongs to a long(1) or short(0) trip or neither(2)
    
    for fly in tqdm(perRunDf.fly.unique()):
        
        numvisits_fly = int(np.amax(perRunDf[perRunDf.fly==fly].after_which_visit.values))
#         numvisits_fly = int(perRunDf[perRunDf.fly==fly].after_which_visit.values[-1])
        
        for visitIndx in range(numvisits_fly+1):
            perRunDf_aftervisit = perRunDf.loc[(perRunDf.fly==fly) & (perRunDf.after_which_visit==visitIndx)]
            numruns = len(perRunDf_aftervisit)
            perTripDf_rel = per_trip_df.loc[(per_trip_df.fly==fly) & (per_trip_df.which_trip==visitIndx)]
            
            if len(perTripDf_rel) > 0:
                triptype_rel = perTripDf_rel.triptype.values[0]
            else:
                triptype_rel = 2 
            
            triptypes.append(triptype_rel*np.ones(numruns))
            
    perRunDf['triptype'] = np.hstack(triptypes)

    return perRunDf

# Augment run DF with trip properties:
# In this version, we allow augmenting an arbitrary set of trip properties not restricted to trip type
def AugmentRunDF_tripProps_v2(perRunDf, per_trip_df, qOIs, nanReplacements):
    triptypes = [] # whether run belongs to a long(1) or short(0) trip or neither(2)

    numqOIs = len(qOIs)
    storagedict = {}
    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        storagedict[qName] = []
    
    
    for fly in tqdm(perRunDf.fly.unique()):
        
        numvisits_fly = int(np.amax(perRunDf[perRunDf.fly==fly].after_which_visit.values))
#         numvisits_fly = int(perRunDf[perRunDf.fly==fly].after_which_visit.values[-1])
        
        for visitIndx in range(numvisits_fly+1):
            perRunDf_aftervisit = perRunDf.loc[(perRunDf.fly==fly) & (perRunDf.after_which_visit==visitIndx)]
            numruns = len(perRunDf_aftervisit)
            perTripDf_rel = per_trip_df.loc[(per_trip_df.fly==fly) & (per_trip_df.which_trip==visitIndx)]

            for qIndx in range(numqOIs):
                qName = qOIs[qIndx]
                if len(perTripDf_rel) > 0:                
                    q_rel = perTripDf_rel[qName].values[0]
                else:
                    q_rel = nanReplacements[qIndx] 
                
                storagedict[qName].append(q_rel*np.ones(numruns))

    for qIndx in range(numqOIs):
        qName = qOIs[qIndx]
        perRunDf[qName] = np.hstack(storagedict[qName])
    
    return perRunDf


# Augment Trip DF with:
# - number of runs during trip
# - average run duration during trip
# - average run velocity during trip (this is the mean of velocities of all runs, rather the mean across all time during trip)
# - average run length (and log10(RL)) during trip
# - std of log10(run length distribution) during trip
# - average turn angle (and log10(turnangle)) during trip
# - std of log10(turn angle distribution) during trip
# - turn bias during trip (max(#CW turns,#antiCW turns)/total#turns)
# - CWfrac during trip (#CWturns/total#turns)
# - covariance between run length and turn angle  
def AugmentTripDF_RunTurnProps(per_trip_df, perRunDf):
    
    numRuns = []
    meanRunDuration = []
    meanRunVelo = []
    meanRL = []
    meanTurnAngle = []
    meanLog10RL = []
    meanLog10TurnAngle = []
    sigmaLog10RL = []
    sigmaLog10TurnAngle = []
    corrLog10RLLog10turn = []
    turnbias = []
    CWfrac = []
    # to see if there's a difference between turns in preferred direction from non-preferred direction: x_bias = (<x|preferred turns> - <x|non-preferred turns>)/<x>
    Log10RL_bias = []
    Log10TurnAngle_bias = []
    RL_bias = []
    TurnAngle_bias = []
    
    for fly in tqdm(per_trip_df.fly.unique()):

        numtrips_fly = np.amax(per_trip_df[per_trip_df.fly==fly].which_trip.values)
        
        for tripIndx in range(numtrips_fly):
            perRunDf_rel = perRunDf.loc[(perRunDf.fly==fly) & (perRunDf.after_which_visit==tripIndx+1) &
                                       (perRunDf.seg_state>=2) & (perRunDf.seg_state<=5)]
            
            runlengths_rel = perRunDf_rel.run_length.values
            rundurations_rel = perRunDf_rel.run_duration.values
            velo_rel = perRunDf_rel.velo.values
            turnangles_rel = perRunDf_rel.turn_angle_fromprev.values
            turnangles_rel = turnangles_rel[runlengths_rel>0]
            runlengths_rel = runlengths_rel[runlengths_rel>0]
            turnangles_rel = turnangles_rel[~np.isnan(runlengths_rel)]
            runlengths_rel = runlengths_rel[~np.isnan(runlengths_rel)]
            absturnangles_rel = np.absolute(turnangles_rel)
            
            # number of runs on this trip
            totnumruns = len(perRunDf_rel)
            numRuns.append(totnumruns)
            
            if totnumruns > 0:
                log10RL_rel = np.log10(runlengths_rel)
                log10turnangles_rel = np.log10(absturnangles_rel)
                if totnumruns > 2:
                    # covariance matrix between log10(runlength) and log10(turnangle)
                    covMat = np.cov(np.array([log10RL_rel,log10turnangles_rel]));
                    sigma0 = np.sqrt(covMat[0,0])
                    sigma1 = np.sqrt(covMat[1,1])
                    if sigma0*sigma1 > 0:
                        corrcoeff = covMat[0,1]/(sigma0*sigma1)
                    else:
                        corrcoeff = np.nan
                else:
                    sigma0 = np.nan
                    sigma1 = np.nan
                    corrcoeff = np.nan

                # number of CW turns
                numCWturns = np.sum(turnangles_rel<0)
                CWfrac_trip = numCWturns/totnumruns
                # turnbias_trip = np.maximum(CWfrac_trip,1.0-CWfrac_trip)
                turnbias_trip = 2*np.maximum(CWfrac_trip,1.0-CWfrac_trip)-1
                
                # means
                meanRL_rel = np.nanmean(runlengths_rel)
                meanturnangle_rel = np.nanmean(absturnangles_rel)
                meanlog10RL_rel = np.nanmean(log10RL_rel)
                meanlog10turnangle_rel = np.nanmean(log10turnangles_rel)
                
                # mean log10RL and log10turnangles among turns in preferred direction
                RL_CW = runlengths_rel[turnangles_rel<0]
                RL_antiCW = runlengths_rel[turnangles_rel>0]
                log10RL_CW = log10RL_rel[turnangles_rel<0]
                log10RL_antiCW = log10RL_rel[turnangles_rel>0]
                turnangles_CW = absturnangles_rel[turnangles_rel<0]
                turnangles_antiCW = absturnangles_rel[turnangles_rel>0]
                log10turnangles_CW = log10turnangles_rel[turnangles_rel<0]
                log10turnangles_antiCW = log10turnangles_rel[turnangles_rel>0]
                if CWfrac_trip > 0.5:
                    RL_bias_rel = (np.nanmean(RL_CW)-np.nanmean(RL_antiCW))/meanRL_rel
                    Log10RL_bias_rel = (np.nanmean(log10RL_CW)-np.nanmean(log10RL_antiCW))/meanlog10RL_rel
                    turnangle_bias_rel = (np.nanmean(turnangles_CW)-np.nanmean(turnangles_antiCW))/meanturnangle_rel
                    log10turnangles_bias_rel = (np.nanmean(log10turnangles_CW)-np.nanmean(log10turnangles_antiCW))/meanlog10turnangle_rel
                else:
                    RL_bias_rel = (np.nanmean(RL_antiCW)-np.nanmean(RL_CW))/meanRL_rel
                    Log10RL_bias_rel = (np.nanmean(log10RL_antiCW)-np.nanmean(log10RL_CW))/meanlog10RL_rel
                    turnangle_bias_rel = (np.nanmean(turnangles_antiCW)-np.nanmean(turnangles_CW))/meanturnangle_rel
                    log10turnangles_bias_rel = (np.nanmean(log10turnangles_antiCW)-np.nanmean(log10turnangles_CW))/meanlog10turnangle_rel
                    
                # append desired quantities
                meanRunDuration.append(np.nanmean(rundurations_rel))
                meanRunVelo.append(np.nanmean(velo_rel))
                meanRL.append(meanRL_rel)
                meanTurnAngle.append(meanturnangle_rel)
                meanLog10RL.append(meanlog10RL_rel)
                meanLog10TurnAngle.append(meanlog10turnangle_rel)
                sigmaLog10RL.append(sigma0)
                sigmaLog10TurnAngle.append(sigma1)
                corrLog10RLLog10turn.append(corrcoeff)
                CWfrac.append(CWfrac_trip)
                turnbias.append(turnbias_trip)
                Log10RL_bias.append(Log10RL_bias_rel)
                Log10TurnAngle_bias.append(log10turnangles_bias_rel)
                RL_bias.append(RL_bias_rel)
                TurnAngle_bias.append(turnangle_bias_rel)
    
            else:
                meanRunDuration.append(np.nan)
                meanRunVelo.append(np.nan)
                meanRL.append(np.nan)
                meanTurnAngle.append(np.nan)
                meanLog10RL.append(np.nan)
                meanLog10TurnAngle.append(np.nan)
                sigmaLog10RL.append(np.nan)
                sigmaLog10TurnAngle.append(np.nan)
                corrLog10RLLog10turn.append(np.nan)
                CWfrac.append(np.nan)
                turnbias.append(np.nan)
                Log10RL_bias.append(np.nan)
                Log10TurnAngle_bias.append(np.nan)
                RL_bias.append(np.nan)
                TurnAngle_bias.append(np.nan)
    

    per_trip_df['numRuns'] = np.hstack(numRuns)
    per_trip_df['meanRunDuration'] = np.hstack(meanRunDuration)
    per_trip_df['meanRunVelo'] = np.hstack(meanRunVelo)
    per_trip_df['meanRL'] = np.hstack(meanRL)
    per_trip_df['meanTurnAngle'] = np.hstack(meanTurnAngle)
    per_trip_df['meanLog10RL'] = np.hstack(meanLog10RL)
    per_trip_df['meanLog10TurnAngle'] = np.hstack(meanLog10TurnAngle)
    per_trip_df['sigmaLog10RL'] = np.hstack(sigmaLog10RL)
    per_trip_df['sigmaLog10TurnAngle'] = np.hstack(sigmaLog10TurnAngle)
    per_trip_df['corrLog10RLLog10turn'] = np.hstack(corrLog10RLLog10turn)
    per_trip_df['turnbias'] = np.hstack(turnbias)
    per_trip_df['CWfrac'] = np.hstack(CWfrac)
    per_trip_df['RL_bias'] = np.hstack(RL_bias)
    per_trip_df['TurnAngle_bias'] = np.hstack(TurnAngle_bias)
    per_trip_df['Log10RL_bias'] = np.hstack(Log10RL_bias)
    per_trip_df['Log10TurnAngle_bias'] = np.hstack(Log10TurnAngle_bias)
    
    
    return per_trip_df

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


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:02:26 2022

@author: labadmin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit
from helper_functions import rle
from helper_functions import fitGMM
from helper_functions import mix_norm_cdf
from helper_functions import roundToSigFig
from CreateNewDataFrames_singlespot import CreatePreturnDF
from statannotations.Annotator import Annotator
from statsmodels.stats.stattools import medcouple

# for plotting autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf
from scipy.interpolate import interp1d

# for plotting
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42

# function for plotting segments of trajectory corresponding to a certain type
# inputs:
# - tmintype: '0' (default) or 'firsthit'
# - tmaxtype: 'max' (default) or 'CFFmax' or 'firsthit'
# - xylimType: 'none' (default) or 'around_food'
def plotTrajSegs(df, segtypeOI, totnumconds, maxnumflies, metadata, body_part = 'body',
                 tmintype = '0', tmaxtype = 'max', colorby = 'time', mksize = 4,
                 ifplotlines = False, ifplotfood = True, ifplotarena = True, ifequal = False,
                 xylimType = 'none', multipleGenotypes = False, fn2save = None):
    names_cond = df.condition.unique()
    f,ax=plt.subplots(maxnumflies,totnumconds,figsize=(totnumconds*5,maxnumflies*6))
    for condIndx in range(totnumconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]
            
            if multipleGenotypes == False:
                gtype = ''
            else:
                gtype = rel_df.genotype.values[0] + ','

            # location and properties of food spot as well as arena size
            food_x = metadata[fly]['arena']['spots']['x']
            food_y = metadata[fly]['arena']['spots']['y']
            scale = metadata[fly]['arena']['scale']
            food_radius = metadata[fly]['arena']['spots']['radius'] /scale
            arena_radius = metadata[fly]['arena']['radius'] /scale
            angle = np.linspace( 0 , 2 * np.pi , 150 )
            if ifplotfood == True:
                foodcircle_x = food_radius * np.cos( angle ) + food_x
                foodcircle_y = food_radius * np.sin( angle ) + food_y
            if ifplotarena == True:
                arenacircle_x = arena_radius * np.cos( angle ) + food_x
                arenacircle_y = arena_radius * np.sin( angle ) + food_y

            if (len(rel_df) > 0) & (np.max(rel_df.CFF.values)>0):
                
                if tmintype == '0':
                    tmin = 0
                elif tmintype == 'firsthit':
#                     Indx_firsthit = np.nonzero(rel_df.CFF.values)[0][0]
                    Indx_firsthit = np.nonzero(rel_df.segment.values==1)[0][0]
                    tmin = rel_df.time.values[Indx_firsthit]
            
                if tmaxtype == 'max':
                    tmax = rel_df.time.values[-1]
                elif tmaxtype == 'CFFmax':
                    Indx_CFFmax = np.argmax(rel_df.CFF.values)
                    if Indx_CFFmax > 0:
                        tmax = rel_df.time.values[Indx_CFFmax]
                    else:
                        tmax = 0
                elif tmaxtype == 'firsthit':
                    Indx_firsthit = np.nonzero(rel_df.segment.values==1)[0][0]
#                     Indx_firsthit = np.nonzero(rel_df.CFF.values)[0][0]
                    tmax = rel_df.time.values[Indx_firsthit]
            
                rel_df = rel_df.loc[(rel_df.time>=tmin)&(rel_df.time<=tmax)]
            
                if len(rel_df) > 0:

                    if totnumconds > 1:
                        axcurr = ax[flyIndx,condIndx]
                    elif maxnumflies > 1:
                        axcurr = ax[flyIndx]
                    else:
                        axcurr = ax
                    # xpositions and ypositions
                    if body_part == 'body':
                        xpos = rel_df.body_x.values
                        ypos = rel_df.body_y.values
                    elif body_part == 'head':
                        xpos = rel_df.head_x.values
                        ypos = rel_df.head_y.values

                    # frames just before animal leaves food source
                    segs_all = rel_df.segment.values
                    toplot = np.isin(segs_all,segtypeOI)

                    if np.max(toplot) > 0:
                        # extract (border->food) or (border->border) or (stay at border) segments during this period
                        if ifplotlines == True:
                            runlen, pos, state, dur = rle(segs_all, dt=rel_df.dt.values) # see function def above
                            if np.max(np.isin(state,segtypeOI)) > 0:
                                pos_rel = pos[np.isin(state,segtypeOI)]
                                runlen_rel = runlen[np.isin(state,segtypeOI)]
                                for kk in range(len(pos_rel)):
                                    axcurr.plot(xpos[pos_rel[kk]:(pos_rel[kk]+runlen_rel[kk])],
                                                              ypos[pos_rel[kk]:(pos_rel[kk]+runlen_rel[kk])], 
                                                              color = str((kk+1)/(len(pos_rel)+1)))

                        colorVec = rel_df[colorby].values[toplot]
                        if (colorby == 'ethogram') | (colorby == 'segment'):
                            colormin = 0
                            # colormax = np.max(df[colorby].values)
                            if (colorby == 'ethogram'):
                                colormax = 4
                            else:
                                colormax = 5
                        else:
                            colormin = np.min(colorVec)
                            colormax = np.max(colorVec)
                        axcurr.scatter(xpos[toplot], ypos[toplot], 
                                                     c = colorVec, 
                                                     s = mksize, vmin = colormin, vmax = colormax, cmap='rainbow')
                        if ifplotfood == True:
                            axcurr.plot(foodcircle_x,foodcircle_y,'k')
                        if ifplotarena == True:
                            axcurr.plot(arenacircle_x,arenacircle_y,'k')
                        axcurr.set_ylabel("y")
                        axcurr.set_xlabel("x")
                        if xylimType != 'none':
                            if xylimType == 'around_food':
                                xlimVec = [food_x-2*food_radius,food_x+2*food_radius]
                                ylimVec = [food_y-2*food_radius,food_y+2*food_radius]
                            axcurr.set_xlim(xlimVec)
                            axcurr.set_ylim(ylimVec)
                        if ifequal == True:
                            axcurr.set_aspect('equal')
            #             f.colorbar(ax)
                        if tmin > 0:
                            axcurr.set_title(gtype + cond + '(fly' + str(flyIndx+1) + 
                                                           ', tmin = ' + str(tmin) + ', tmax = ' + str(round(tmax)) + ')')
                        else:
                            axcurr.set_title(gtype + cond + '(fly' + str(flyIndx+1) + ', tmax = ' + str(round(tmax)) + ')')
            #             axcurr.legend()
    if fn2save != None:
            f.savefig(fn2save, bbox_inches='tight', dpi=300)
        

# function for plotting trajectories of a quantity of interest as a function of time (or other analogous quantity)
# for individual flies on different subplots
def plotTimeTraj(df, qOI, maxnumflies, xOI = 'time', tstart = 'exptstart', tend = 'exptend',
                 ifdiffq = False, iflogq = False, ifacf = False, groupOI = None, 
                 indvar = None, iflogInd = False, crosstype = None, ifoverlap = False,
                 mksize = 1, crossSize = 20, multipleGenotypes = False, colwidth = 5, rowwidth = 6,
                 fn2save = None):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    if iflogq == False:
        qname = qOI
    else:
        qname = 'log10(' + qOI + ')'
    if ifdiffq == True:
        qname = 'd(' + qname + ')'
    
    if ifoverlap == False:
        numrows = maxnumflies
    else:
        if multipleGenotypes == True:
            names_Genotypes = df.genotype.unique()
            numGenotypes = len(names_Genotypes)
        else:
            numGenotypes = 1
        numrows = numGenotypes
    
    if ifacf == False:
        numcols = numconds
    else:
        numcols = numconds*2
    f,ax=plt.subplots(numrows,numcols,figsize=(numcols*colwidth,numrows*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]
            
            if 'time' in rel_df.columns:
                if tstart == 'exptstart':
                    t0 = 0
                elif tstart == 'foodstart':
                    if np.sum(rel_df.segment.values == 1) > 0:
                        t0 = rel_df.time.values[np.where(rel_df.segment.values == 1)[0][0]]
                    else:
                        t0 = np.max(rel_df.time.values)
                if tend == 'exptend':
                    t1 = np.max(rel_df.time.values)
                elif tend == 'foodstart':
                    if np.sum(rel_df.segment.values == 1) > 0:
                        t1 = rel_df.time.values[np.where(rel_df.segment.values == 1)[0][0]]
                    else:
                        t1 = np.max(rel_df.time.values)
                elif tend == 'wallstart_afterfirstvisit':
                    if np.sum(rel_df.segment.values == 1) > 0:
                        tv = rel_df.time.values[np.where(rel_df.segment.values == 1)[0][0]]
                    else:
                        tv = np.max(rel_df.time.values)
                    posInds = np.where((rel_df.segment.values == 0) & (rel_df.time.values > tv))
                    if len(posInds[0]) > 0:
                        t1 = rel_df.time.values[posInds[0][0]]
                    else:
                        t1 = np.max(rel_df.time.values)
                rel_df = rel_df.loc[(rel_df.time > t0) & (rel_df.time < t1)]
            
            if len(rel_df) > 0:
                
                if multipleGenotypes == False:
                    gtype = ''
                else:
                    gtype = rel_df.genotype.values[0] + ','
                
                if ifacf == False:
                    colIndx = condIndx
                else:
                    colIndx = condIndx*2
                if ifoverlap == False:
                    axcurr = ax[flyIndx,colIndx]
                    if ifacf == True:
                        axacf = ax[flyIndx,colIndx+1]
                    titlename = gtype + cond + '(fly' + str(flyIndx+1) + ')'                    
                        
                else:
                    titlename = gtype + cond
                    if (multipleGenotypes == True) & (numGenotypes > 1):
                        whichgenotype = np.where(names_Genotypes == rel_df.genotype.values[0])[0][0]
                        axcurr = ax[whichgenotype,colIndx]
                        if ifacf == True:
                            axacf = ax[whichgenotype,colIndx+1]
                    else:
                        axcurr= ax[colIndx]
                        if ifacf == True:
                            axacf = ax[colIndx+1]
                
                if groupOI != None:
                    groups = rel_df.groupby(groupOI)
                    for name, group in groups:
                        xVec = group[xOI]
                        qVec = group[qOI]
                        if iflogq == True:
                            xVec = xVec[qVec>0]
                            qVec = np.log10(qVec[qVec>0])
                        if ifdiffq == True:
                            qVec = np.diff(qVec)
                            xVec = xVec[:-1]
                        axcurr.plot(xVec, qVec, marker='o', linestyle='', markersize=mksize, 
                                                  alpha = 0.5, label = groupOI + '=' +  str(name))
                else:
                    xVec = rel_df[xOI].values
                    qVec = rel_df[qOI].values
                    if indvar != None:
                        indVec = rel_df[indvar].values
                        if iflogInd == True:
                            indVec = mksize*np.log10(1+indVec)
                        sVec = indVec + mksize
                    else:
                        sVec = np.ones(len(xVec))*mksize
                    xVec = xVec[~np.isnan(sVec)]
                    qVec = qVec[~np.isnan(sVec)]
                    sVec = sVec[~np.isnan(sVec)]
                    if iflogq == True:
                        if indvar != None:
                            sVec = sVec[qVec>0]
                        xVec = xVec[qVec>0]
                        qVec = np.log10(qVec[qVec>0])
                        
                    if ifdiffq == True:
                        qVec = np.diff(qVec)
                        xVec = xVec[:-1]
                        if indvar != None:
                            sVec = sVec[:-1]
                    axcurr.plot(xVec, qVec)
                    # ax[flyIndx,condIndx].plot(xVec, qVec, marker='o', linestyle='', markersize=mksize, alpha = 0.5)
                    axcurr.scatter(xVec, qVec, s = sVec, marker = 'o', alpha = 0.5)
                    # ax[flyIndx,condIndx].scatter(xVec, qVec, c = sVec, s = sizeVec, 
                    #                      vmin = colormin, vmax = colormax, cmap='rainbow')
                    
                    if (ifacf == True) & (np.sum(~np.isnan(xVec))>0):
                        qVec = qVec[~np.isnan(xVec)]
                        xVec = xVec[~np.isnan(xVec)]
                        xVec = xVec[~np.isnan(qVec)]
                        qVec = qVec[~np.isnan(qVec)]
                        if len(xVec) > 500:
                            xnew = np.arange(np.nanmin(xVec),np.nanmax(xVec),1.0)
                            qnew = interp1d(xVec,qVec)(xnew)
                            if len(xnew) > 800:
                                # plot_acf(qnew,ax = axacf)
                                plot_acf(qnew,ax = axacf, lags = 500)
                                axacf.set_ylabel('autocorrelation')
                                axacf.set_xlabel('lag')

                        
                if ifdiffq == False:
                    if crosstype == 'tripstart':
                        # when fly first starts a trip
                        segs_all = rel_df.segment.values
                        # frames just before animal leaves food source
                        frameInds_beforeleaving = np.where(np.array((segs_all[:-1] == 1) & (segs_all[1:] != segs_all[:-1])))[0] 
                        tcrosses = rel_df[xOI].values[frameInds_beforeleaving]
                        qcrosses = rel_df[qOI].values[frameInds_beforeleaving]
                    elif crosstype == 'feedend':
                        # when fly first stops feeding
                        ethos = rel_df.ethogram.values
                        # frames just before animal stops feeding
                        frameInds_endfeed = np.where(np.array((ethos[:-1] == 3) & (ethos[1:] != ethos[:-1])))[0] 
                        tcrosses = rel_df[xOI].values[frameInds_endfeed]
                        qcrosses = rel_df[qOI].values[frameInds_endfeed]
                    elif crosstype != None:
                        frameInds = np.where(rel_df[crosstype].values == 1)[0]
                        tcrosses = rel_df[xOI].values[frameInds]
                        qcrosses = rel_df[qOI].values[frameInds]
                    if crosstype != None:
                        if iflogq == True:
                            tcrosses = tcrosses[qcrosses>0]
                            qcrosses = np.log10(qcrosses[qcrosses>0])
                        axcurr.scatter(tcrosses, qcrosses, s = crossSize, marker = 'x', c = 'black')
                
                axcurr.set_ylabel(qname)
                axcurr.set_xlabel(xOI)

                axcurr.set_title(titlename)
                if (groupOI != None) & (ifoverlap == False):
                    axcurr.legend()
        if fn2save != None:
            f.savefig(fn2save, bbox_inches='tight', dpi=300)
             
            
# function for plotting trajectories of a quantity of interest as a function of time (or other analogous quantity)
# for individual flies on the same plot
def plotTimeTraj_allflies(df, qOI, xOI = 'time', iflogq = False, crosstype = None, 
                          mksize = 18 , ybounds = 'none', multipleGenotypes = False, colwidth = 5, rowwidth = 6):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    if iflogq == False:
        qname = qOI
    else:
        qname = 'log10(' + qOI + ')'
        
    if multipleGenotypes == False:
        numGenotypes = 1
    else:
        names_Genotypes = df.genotype.unique()
        numGenotypes = len(names_Genotypes)
    numrows = 2*numGenotypes
    f,ax=plt.subplots(numrows,numconds,figsize=(numconds*colwidth,numrows*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        
        for genotypeIndx in range(numGenotypes):
            if multipleGenotypes == True:
                genotype_df = perCond_df.loc[perCond_df.genotype == names_Genotypes[genotypeIndx]]
                title = names_Genotypes[genotypeIndx] + ',' + cond
            else:
                genotype_df = perCond_df
                title = cond
            
            for flyIndx in range(len(genotype_df.fly.unique())):
                fly = genotype_df.fly.unique()[flyIndx]
                rel_df = genotype_df.loc[genotype_df.fly==fly]
                if len(rel_df) > 0:
                    xVec = rel_df[xOI].values
                    qVec = rel_df[qOI].values
                    if iflogq == True:
                        xVec = xVec[qVec>0]
                        qVec = np.log10(qVec[qVec>0])
                        
                    ax[genotypeIndx*2,condIndx].plot(xVec, qVec)
                    if crosstype == 'tripstart':
                        # when fly first starts a trip
                        segs_all = rel_df.segment.values
                        # frames just before animal leaves food source
                        frameInds_beforeleaving = np.where(np.array((segs_all[:-1] == 1) & (segs_all[1:] != segs_all[:-1])))[0] 
                        tcrosses = rel_df[xOI].values[frameInds_beforeleaving]
                        qcrosses = rel_df[qOI].values[frameInds_beforeleaving]
                    elif crosstype == 'feedend':
                        # when fly first stops feeding
                        ethos = rel_df.ethogram.values
                        # frames just before animal stops feeding
                        frameInds_endfeed = np.where(np.array((ethos[:-1] == 3) & (ethos[1:] != ethos[:-1])))[0] 
                        tcrosses = rel_df[xOI].values[frameInds_endfeed]
                        qcrosses = rel_df[qOI].values[frameInds_endfeed]
                    elif crosstype != None:
                        frameInds = np.where(rel_df[crosstype].values == 1)[0]
                        tcrosses = rel_df[xOI].values[frameInds]
                        qcrosses = rel_df[qOI].values[frameInds]
                    
                    if crosstype != None:
                        if iflogq == True:
                            tcrosses = tcrosses[qcrosses>0]
                            qcrosses = np.log10(qcrosses[qcrosses>0])
                        ax[genotypeIndx*2,condIndx].scatter(tcrosses, qcrosses, s = mksize, marker = 'x')
                                
                    if np.max(rel_df.CFF.values)>0:
                        firstfeedtime = rel_df[xOI][rel_df.CFF.values>0].values[0]
                        tshifted = xVec - firstfeedtime
                        qVec_shifted = qVec[tshifted>=0]
                        tshifted = tshifted[tshifted>=0]
                        ax[genotypeIndx*2+1,condIndx].plot(tshifted, qVec_shifted)
                        if crosstype != None:
                            tcrosses_shifted = tcrosses - firstfeedtime
                            qcrosses_shifted = qcrosses[tcrosses_shifted>=0]
                            tcrosses_shifted = tcrosses_shifted[tcrosses_shifted>=0]
                            ax[genotypeIndx*2+1,condIndx].scatter(tcrosses_shifted, qcrosses_shifted, s = mksize, marker = 'x')
                        
            ax[genotypeIndx*2,condIndx].set_title(title)
            ax[genotypeIndx*2,condIndx].set_ylabel(qname)
            ax[genotypeIndx*2,condIndx].set_xlabel(xOI)
            ax[genotypeIndx*2+1,condIndx].set_ylabel(qname)
            ax[genotypeIndx*2+1,condIndx].set_xlabel(xOI + " from first feeding")
            if ybounds != 'none':
                ax[genotypeIndx*2,condIndx].set_ylim(ybounds[condIndx])
                ax[genotypeIndx*2+1,condIndx].set_ylim(ybounds[condIndx])

# function for plotting trajectories of a quantity of interest as a function of time (or other analogous quantity)
# for individual trips or visits of each fly 
# Inputs:
# - segOI: 'trip' or 'visit' (old version)
# - groupOI: 'whichtrip' or 'whichvisit'
def plotTimeTraj_indivTripsOrVisits(df, qOI, xOI, maxnumflies, groupOI = 'whichtrip', iflogq = False, iflogx = False,
                                    numderiv_q = 0, mksize = 18 , multipleGenotypes = False, colwidth = 5, rowwidth = 6):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    if iflogq == False:
        qname = qOI
    else:
        qname = 'log10(' + qOI + ')'
    if iflogx == False:
        xname = xOI
    else:
        xname = 'log10(' + xOI + ')'
        
    if numderiv_q > 0:
        qname = 'd' + str(numderiv_q) + qname
        
    # if segOI == 'trip':
    #     groupOI = 'whichtrip'
    # elif segOI == 'visit':
    #     groupOI = 'whichvisit'
        
    numrows = maxnumflies
    f,ax=plt.subplots(numrows,numconds,figsize=(numconds*colwidth,numrows*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]
                
            if len(rel_df) > 0:
                
                if multipleGenotypes == False:
                    gtype = ''
                else:
                    gtype = rel_df.genotype.values[0] + ','
                
                axcurr = ax[flyIndx,condIndx]
                titlename = gtype + cond + '(fly' + str(flyIndx+1) + ')'                    
                        
                groups = rel_df.groupby(groupOI)
                for name, group in groups:
                    xVec = group[xOI].values
                    qVec = group[qOI].values

                    if len(xVec) > numderiv_q:        
                    
                        if iflogq == True:
                            xVec = xVec[qVec>0]
                            qVec = np.log10(qVec[qVec>0])
    
                        if iflogx == True:
                            xVec = np.log10(xVec)

                        if numderiv_q > 0:
                            for derivIndx in range(numderiv_q):
                                qVec = np.diff(qVec)
                                xVec = (xVec[0:-1] + xVec[1:])/2

                        axcurr.plot(xVec, qVec, marker='o', linestyle='', markersize=mksize, 
                                                  alpha = 0.5, label = groupOI + '=' +  str(name))
                    
                        
                axcurr.set_ylabel(qname)
                axcurr.set_xlabel(xname)
                axcurr.set_title(titlename)
                # axcurr.legend()
            


# function for comparing properties of flies across different conditions and genotypes
# TO DO: statistical significance between different genotypes:
# https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
def boxplot_compareConds(df, qOIvec, maxnumcols = 4, rowwidth = 5, colwidth = 5, 
                         multipleGenotypes = False, ifcalstats_genotype = False,
                         stattest = "Mann-Whitney",printstats = False, fn2save = None):
    
    numqs = len(qOIvec)
    numcols = np.maximum(int(np.floor(np.sqrt(numqs))),maxnumcols)
    numrows = int(np.ceil(numqs/numcols))
#     numrows = int(np.floor(np.sqrt(numqs)))
#     numcols = int(np.ceil(numqs/numrows))

    names_cond = df.condition.unique()
    if ifcalstats_genotype == True:
        pairs = []
        for condIndx in range(len(names_cond)):
            cond = names_cond[condIndx]
            conddf = df.loc[df.condition == cond]
            gnames = conddf.genotype.unique()
            nullmodel = (cond,gnames[0])
            if len(gnames) > 1:
                for gIndx in range(len(gnames)-1):
                    model2compare = (cond,gnames[gIndx+1])
                    pairs.append([nullmodel,model2compare])
                    
    # if iflogvec == False:
    #     iflogvec = [False]*numqs
        
    f,axes=plt.subplots(numrows,numcols,figsize=(numcols*colwidth,numrows*rowwidth))
    for qIndx in range(numqs):
        rowIndx = int(np.floor(qIndx/numcols))
        colIndx = qIndx - rowIndx*numcols
        qOI = qOIvec[qIndx]
        
        if (numrows>1) & (numcols>1):
            currax = axes[rowIndx,colIndx]
        else:
            currax = axes[colIndx]
        
        if multipleGenotypes == False:
            sns.boxplot(data = df, x = 'condition', y = qOI, showfliers = False, ax = currax)
        else:
            sns.boxplot(data = df, x = 'condition', y = qOI, hue = 'genotype', showfliers = False,
                       ax = currax)
            if ifcalstats_genotype == True:
                annotator = Annotator(currax, pairs, 
                                      data = df, x = 'condition', y = qOI, hue = 'genotype')
                annotator.configure(test=stattest,verbose = printstats).apply_and_annotate()
        # if iflogvec[qIndx] == True:
        #     currax.set_yscale("log")
            
        currax.set_xticklabels(names_cond,)
        currax.set_ylabel(qOI)
        
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)


# function for plotting boxplot showing the distribution of a specified quantity of interest as a function of 
# another specified x variable across all flies for different experimental conditions.
# In this v3, we allow plot all conditions on the same figure (common x axis)
def plot_boxplot_xcontinuous_v3(df,qOI,xVars,xBinEdges_all,ifshowNsamples = False,ifcalstats = False,iflogx = False,
                             iflogq = False, xlabeltype = 'binmid', rounding = [2], colwidth = 5, rowwidth = 5,
                             condcolors = {'0-125M_24hr': 'orangered', '0-125M_40hr': 'brown','0M_24hr': 'lightseagreen','0M_40hr':'teal'},
                             xlabelname = 'xGroup', setylim = None, auto_xBin = False, autoNumxbins = 4, testtrend = False,
                             stattest = "Mann-Whitney", correctiontype = None, printstats = False, fn2save = None):
    names_genotype = df.genotype.unique()
    numgenotypes = len(names_genotype)
    numrows = len(xVars)
    if len(rounding) == 1:
        rounding = [rounding[0]]*numrows
    if ifshowNsamples == False:
        ifshowNsamples = [False]*numrows
    if ifcalstats == False:
        ifcalstats = [False]*numrows
    if iflogx == False:
        iflogx = [False]*numrows
    if xlabeltype == 'binmid':
        xlabeltype = ['binmid']*numrows
    elif xlabeltype == 'binrange':
        xlabeltype = ['binrange']*numrows
    
    if iflogq == False:
        yname = qOI
    else:
        yname = 'log$_{10}$(' + qOI + ')'
    
    f,ax=plt.subplots(numrows,numgenotypes,figsize=(numgenotypes*colwidth,numrows*rowwidth))
    for rowIndx in range(numrows):
        xVar = xVars[rowIndx]
        if auto_xBin == False:
            xBinEdges_row = xBinEdges_all[rowIndx]
            if (len(xBinEdges_row)==1) and (numgenotypes>1):
                xBinEdges_row = [xBinEdges_row]*numgenotypes
        for genotypeIndx in range(numgenotypes):
            genotype = names_genotype[genotypeIndx]
            perGenotype_df = df.loc[(df.genotype == genotype)]
            
            condVec_all = perGenotype_df['condition'].values
            condnames = perGenotype_df.condition.unique()
            
            qVec_all = perGenotype_df[qOI].values
            xvals_all = perGenotype_df[xVar].values
            if iflogq == True:
                condVec_all = condVec_all[qVec_all>0]
                xvals_all = xvals_all[qVec_all>0]
                qVec_all = np.log10(qVec_all[qVec_all>0])
                    
            if iflogx[rowIndx] == True:
                condVec_all = condVec_all[xvals_all>0]
                qVec_all = qVec_all[xvals_all > 0]
                xvals_all = np.log10(xvals_all[xvals_all>0])
                
            # assign values to bins
            if auto_xBin == True:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            else:
                xBinEdges = xBinEdges_row[genotypeIndx]
                if len(xBinEdges) == 1:
                    xBinEdges = xBinEdges[0]
                numxbins = len(xBinEdges)-1
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            
            bin_indices = np.digitize(xvals_all, xBinEdges)
            binnames = [str(i+1) for i in range(numxbins)]
            xnames = []
            # xname_all = np.zeros(len(xvals_all))
            # xname_all = [None]*len(xvals_all)
            for kk in range(numxbins):
                if xlabeltype[rowIndx] == 'binmid':
                    if rounding[rowIndx] > 0:
                        xname = str(round(xBinMid[kk],rounding[rowIndx]))
                    else:
                        xname = str(round(xBinMid[kk]))
                elif xlabeltype[rowIndx] == 'binrange':
                    if rounding[rowIndx] > 0:
                        xname = str(round(xBinEdges[kk],rounding[rowIndx])) + '-' + str(round(xBinEdges[kk+1],rounding[rowIndx]))
                    else:
                        xname = str(round(xBinEdges[kk])) + '-' + str(round(xBinEdges[kk+1]))
                xnames.append(xname)
                # xname_all[bin_indices==(kk+1)] = xname
            
            # Create temporary dataframe
            df_data = {xlabelname: [str(x) for x in bin_indices],
                            'qval': qVec_all,
                            'condition':condVec_all}
            temp_df_data = pd.DataFrame(df_data)
            if ifshowNsamples[rowIndx] == True: # show number of data samples in each xbin
                nsamplesVec = []
                medianVec = []
                for kk in range(numxbins):
                    nsamplesVec.append(len(temp_df_data[temp_df_data.xGroup==binnames[kk]]))
                    medianVec.append(np.median(temp_df_data[temp_df_data.xGroup == binnames[kk]].qval.values))
                nobs = [str(x) for x in nsamplesVec]
                nobs = ["n: " + i for i in nobs]
                
            if ifcalstats[rowIndx] == True: # calculate statistical significance 
                pairs = []
                # # between neighboring x distributions
                # for xbinIndx in range(numxbins-1):
                #     if (multipleGenotypes == True) & (len(gnames)>1):
                #         for gIndx in range(len(gnames)):
                #             pairs.append([(binnames[xbinIndx],gnames[gIndx]),(binnames[xbinIndx+1],gnames[gIndx])])
                #     else:
                #         pairs.append([binnames[xbinIndx],binnames[xbinIndx+1]])
                    
                # between first and last x bin:
                # if numxbins > 2:
                for condIndx in range(len(condnames)):
                    pairs.append([(binnames[0],condnames[condIndx]),(binnames[numxbins-1],condnames[condIndx])])
                        
                # between conditions for the same x bin:
                for xbinIndx in range(numxbins):
                    nullmodel = (binnames[xbinIndx],gnames[0])
                    for condIndx in range(len(condnames)-1):
                        model2compare = (binnames[xbinIndx],condnames[condIndx+1])
                        pairs.append([nullmodel,model2compare])

            if (numrows > 1) & (numgenotypes > 1):
                axcurr = ax[rowIndx,genotypeIndx]
            elif (numrows == 1) & (numgenotypes == 1):
                axcurr = ax
            elif numrows == 1:
                axcurr = ax[genotypeIndx]
            elif numgenotypes == 1:
                axcurr = ax[rowIndx]
            
            boxplot_params = {'data': temp_df_data, 'x': xlabelname, 'y': 'qval', 
                              'order': binnames} #, showfliers: False
            boxplot_params['hue'] = 'condition'
            boxplot_params['palette'] = condcolors
            
            sns.boxplot(ax = axcurr, **boxplot_params)
            if ifcalstats[rowIndx] == True:
                annotator = Annotator(axcurr, pairs, **boxplot_params)
                annotator.configure(test = stattest,comparisons_correction = correctiontype,verbose = printstats).apply_and_annotate()
            
            # number of data samples
            if ifshowNsamples[rowIndx] == True:
                pos = range(len(nobs))
                for tick,label in zip(pos,axcurr.get_xticklabels()):
                    axcurr.text(pos[tick],
                            medianVec[tick] + 0.03,
                            nobs[tick],
                            horizontalalignment='center',
                            size='medium',
                            color='k',
                            weight='semibold')
            
            # test for trend in data 
            if testtrend == True:
                    # tauVec = []
                    # pVec = []
                text_trend = ''
                for condIndx in range(len(condnames)):
                    tau, p_value = stats.kendalltau(xvals_all[condVec_all == condnames[condIndx]],
                                                    qVec_all[condVec_all == condnames[condIndx]])
                    # tauVec.append(tau)
                    # pVec.append(p_value)
                    text_trend = text_trend + condnames[condIndx] + ': tau = ' + str(roundToSigFig(tau,2)) + ', p = ' + str(roundToSigFig(p_value,2)) + ',       '  
                axcurr.text(1, np.min(qVec_all)+(np.max(qVec_all)-np.min(qVec_all))*0.3,
                            text_trend, ha='left', va='bottom', color='k')
                
                    
            axcurr.set_xticklabels(xnames)
            axcurr.set_ylabel(yname)
            if setylim != None:
                axcurr.set_ylim(setylim[0],setylim[1])
            axcurr.legend(loc = 'best')
#             axcurr.legend(loc = 'upper left')
            if iflogx[rowIndx] == False:
                axcurr.set_title(xVar + '(' + genotype + ')')
            else:
                axcurr.set_title('log10(' + xVar + ') (' + genotype + ')')
                
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)


# function for plotting boxplot showing the distribution of a specified quantity of interest as a function of 
# another specified x variable across all flies for different experimental conditions.
# In this v2, instead of creating boxplot and calculating significance manually, we construct a temporary
# dataframe for the data, use seaborn's boxplot function for plotting and 
# the Annonator package for statistical signifcance.
# We also allow comparison between multiple genotypes.
def plot_boxplot_xcontinuous_v2(df,qOI,xVars,xBinEdges_all,ifshowNsamples = False,ifcalstats = False,iflogx = False,
                             iflogq = False, xlabeltype = 'binmid', rounding = [2], colwidth = 5, rowwidth = 5,
                             auto_xBin = False, autoNumxbins = 4, multipleGenotypes = False, testtrend = False,
                             stattest = "Mann-Whitney", correctiontype = None, printstats = False, fn2save = None):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    numrows = len(xVars)
    if len(rounding) == 1:
        rounding = [rounding[0]]*numrows
    if ifshowNsamples == False:
        ifshowNsamples = [False]*numrows
    if ifcalstats == False:
        ifcalstats = [False]*numrows
    if iflogx == False:
        iflogx = [False]*numrows
    if xlabeltype == 'binmid':
        xlabeltype = ['binmid']*numrows
    elif xlabeltype == 'binrange':
        xlabeltype = ['binrange']*numrows
    if iflogq == False:
        yname = qOI
    else:
        yname = 'log10(' + qOI + ')'
    
    f,ax=plt.subplots(numrows,numconds,figsize=(numconds*colwidth,numrows*rowwidth))
    for rowIndx in range(numrows):
        xVar = xVars[rowIndx]
        if auto_xBin == False:
            xBinEdges_row = xBinEdges_all[rowIndx]
            if (len(xBinEdges_row)==1) and (numconds>1):
                xBinEdges_row = [xBinEdges_row]*numconds
        for condIndx in range(numconds):
            cond = names_cond[condIndx]
            perCond_df = df.loc[(df.condition == cond)]
            
            if multipleGenotypes == True:
                genotypeVec_all = perCond_df['genotype'].values
                gnames = perCond_df.genotype.unique()
            
            qVec_all = perCond_df[qOI].values
            xvals_all = perCond_df[xVar].values
            if iflogq == True:
                if multipleGenotypes == True:
                    genotypeVec_all = genotypeVec_all[qVec_all>0]
                xvals_all = xvals_all[qVec_all>0]
                qVec_all = np.log10(qVec_all[qVec_all>0])
                    
            if iflogx[rowIndx] == True:
                if multipleGenotypes == True:
                    genotypeVec_all = genotypeVec_all[xvals_all>0]
                qVec_all = qVec_all[xvals_all > 0]
                xvals_all = np.log10(xvals_all[xvals_all>0])
                
            # assign values to bins
            if auto_xBin == True:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            else:
                xBinEdges = xBinEdges_row[condIndx]
                if len(xBinEdges) == 1:
                    xBinEdges = xBinEdges[0]
                numxbins = len(xBinEdges)-1
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            
            bin_indices = np.digitize(xvals_all, xBinEdges)
            binnames = [str(i+1) for i in range(numxbins)]
            xnames = []
            # xname_all = np.zeros(len(xvals_all))
            # xname_all = [None]*len(xvals_all)
            for kk in range(numxbins):
                if xlabeltype[rowIndx] == 'binmid':
                    xname = str(round(xBinMid[kk],rounding[rowIndx]))
                elif xlabeltype[rowIndx] == 'binrange':
                    xname = str(round(xBinEdges[kk],rounding[rowIndx])) + '-' + str(round(xBinEdges[kk+1],rounding[rowIndx]))
                xnames.append(xname)
                # xname_all[bin_indices==(kk+1)] = xname
            
            # Create temporary dataframe
            if multipleGenotypes == False:
                df_data = {'xGroup': [str(x) for x in bin_indices],
                                'qval': qVec_all,}
            else:
                df_data = {'xGroup': [str(x) for x in bin_indices],
                                'qval': qVec_all,
                                'genotype':genotypeVec_all}
            temp_df_data = pd.DataFrame(df_data)
            if ifshowNsamples[rowIndx] == True: # show number of data samples in each xbin
                nsamplesVec = []
                medianVec = []
                for kk in range(numxbins):
                    nsamplesVec.append(len(temp_df_data[temp_df_data.xGroup==binnames[kk]]))
                    medianVec.append(np.median(temp_df_data[temp_df_data.xGroup == binnames[kk]].qval.values))
                # nobs = temp_df_data['xGroup'].value_counts().values
                nobs = [str(x) for x in nsamplesVec]
                nobs = ["n: " + i for i in nobs]
                # medianVec = temp_df_data.groupby(['xGroup'])['qval'].median().values
                
            if ifcalstats[rowIndx] == True: # calculate statistical significance 
                pairs = []
                # # between neighboring x distributions
                # for xbinIndx in range(numxbins-1):
                #     if (multipleGenotypes == True) & (len(gnames)>1):
                #         for gIndx in range(len(gnames)):
                #             pairs.append([(binnames[xbinIndx],gnames[gIndx]),(binnames[xbinIndx+1],gnames[gIndx])])
                #     else:
                #         pairs.append([binnames[xbinIndx],binnames[xbinIndx+1]])
                    
                # between first and last x bin:
                # if numxbins > 2:
                if (multipleGenotypes == True) and (len(gnames)>1):
                    for gIndx in range(len(gnames)):
                        pairs.append([(binnames[0],gnames[gIndx]),(binnames[numxbins-1],gnames[gIndx])])
                else:
                    pairs.append([binnames[0],binnames[numxbins-1]])
                        
                # between genotypes for the same x bin:
                if (multipleGenotypes == True) and (len(gnames)>1):
                    for xbinIndx in range(numxbins):
                        nullmodel = (binnames[xbinIndx],gnames[0])
                        for gIndx in range(len(gnames)-1):
                            model2compare = (binnames[xbinIndx],gnames[gIndx+1])
                            pairs.append([nullmodel,model2compare])

            if (numrows > 1) & (numconds > 1):
                axcurr = ax[rowIndx,condIndx]
            elif numrows == 1:
                axcurr = ax[condIndx]
            elif numconds == 1:
                axcurr = ax[rowIndx]
            
            boxplot_params = {'data': temp_df_data, 'x': 'xGroup', 'y': 'qval', 
                              'order': binnames} #, showfliers: False
            if (multipleGenotypes == True) and (len(gnames)>1):
                boxplot_params['hue'] = 'genotype'
            
            sns.boxplot(ax = axcurr, **boxplot_params)
            # if (multipleGenotypes == True) & (len(gnames)>1):
            #     sns.move_legend(axcurr, "lower right")
            if ifcalstats[rowIndx] == True:
                annotator = Annotator(axcurr, pairs, **boxplot_params)
                annotator.configure(test = stattest,comparisons_correction = correctiontype,verbose = printstats).apply_and_annotate()
            
            # number of data samples
            if ifshowNsamples[rowIndx] == True:
                pos = range(len(nobs))
                for tick,label in zip(pos,axcurr.get_xticklabels()):
                    axcurr.text(pos[tick],
                            medianVec[tick] + 0.03,
                            nobs[tick],
                            horizontalalignment='center',
                            size='medium',
                            color='k',
                            weight='semibold')
            
            # test for trend in data 
            if testtrend == True:
                if (multipleGenotypes == True) and (len(gnames)>1):
                    # tauVec = []
                    # pVec = []
                    text_trend = ''
                    for gIndx in range(len(gnames)):
                        tau, p_value = stats.kendalltau(xvals_all[genotypeVec_all == gnames[gIndx]],
                                                        qVec_all[genotypeVec_all == gnames[gIndx]])
                        # tauVec.append(tau)
                        # pVec.append(p_value)
                        text_trend = text_trend + gnames[gIndx] + ': tau = ' + str(roundToSigFig(tau,2)) + ', p = ' + str(roundToSigFig(p_value,2)) + ',       '  
                else:
                    tau, p_value = stats.kendalltau(xvals_all,qVec_all)
                    text_trend = 'tau = ' + str(roundToSigFig(tau,2)) + ', p = ' + str(roundToSigFig(p_value,2)) 
                axcurr.text(1, np.min(qVec_all)+(np.max(qVec_all)-np.min(qVec_all))*0.3,
                            text_trend, ha='left', va='bottom', color='k')
                
                    
            axcurr.set_xticklabels(xnames)
            axcurr.set_ylabel(yname)
            if (multipleGenotypes == True) and (len(gnames)>1):
                axcurr.legend(loc = 'upper left')
            if iflogx[rowIndx] == False:
                axcurr.set_title(xVar + '(' + cond + ')')
            else:
                axcurr.set_title('log10(' + xVar + ') (' + cond + ')')
                
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)

            
# function for plotting boxplot showing the distribution of a specified quantity of interest as a function of 
# another specified x variable across all flies for different experimental conditions.
# This function allows plotting against multiple x variables (which corresponds to the different rows).
def plot_boxplot_xcontinuous(df,qOI,xVars,xBinEdges_all,ifshowNsamples = False,ifcalstats = False,iflogx = False,
                             ifpositivex = True, ifpositiveq = True,
                             iflogq = False,alpha = 0.05,xlabeltype = 'binmid', rounding = [2], colwidth = 5, rowwidth = 5,
                            auto_xBin = False, autoNumxbins = 4, multipleGenotypes = False, fn2save = None):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    numrows = len(xVars)
    if len(rounding) == 1:
        rounding = [rounding[0]]*numrows
    if ifshowNsamples == False:
        ifshowNsamples = [False]*numrows
    if ifcalstats == False:
        ifcalstats = [False]*numrows
    if iflogx == False:
        iflogx = [False]*numrows
    if xlabeltype == 'binmid':
        xlabeltype = ['binmid']*numrows
    if iflogq == False:
        yname = qOI
    else:
        yname = 'log10(' + qOI + ')'
        
    # if multipleGenotypes == False:
    #     numGenotypes = 1
    # else:
    #     names_Genotypes = df.genotypes.unique()
    #     numGenotypes = len(names_Genotypes)
    
    f,ax=plt.subplots(numrows,numconds,figsize=(numconds*colwidth,numrows*rowwidth))
    for rowIndx in range(numrows):
        xVar = xVars[rowIndx]
        if auto_xBin == False:
            xBinEdges_row = xBinEdges_all[rowIndx]
            if (len(xBinEdges_row)==1) & (numconds>1):
                xBinEdges_row = [xBinEdges_row]*numconds
        for condIndx in range(numconds):
            cond = names_cond[condIndx]
            perCond_df = df.loc[(df.condition == cond)]
            
            # if multipleGenotypes == True:
            #     genotypeVec_all = perCond_df[genotype].values
            
            qVec_all = perCond_df[qOI].values
            xvals_all = perCond_df[xVar].values
            if ifpositiveq == True:
                xvals_all = xvals_all[qVec_all>0]
                qVec_all = qVec_all[qVec_all>0]
            if ifpositivex == True:
                qVec_all = qVec_all[xvals_all > 0]
                xvals_all = xvals_all[xvals_all>0]
                
            if iflogq == True:
                xvals_all = xvals_all[qVec_all>0]
                qVec_all = np.log10(qVec_all[qVec_all>0])
                # if multipleGenotypes == True:
                #     genotypeVec_all = genotypeVec_all[qVec_all>0]
                    
            if iflogx[rowIndx] == True:
                qVec_all = qVec_all[xvals_all > 0]
                xvals_all = np.log10(xvals_all[xvals_all>0])
                # if multipleGenotypes == True:
                #     genotypeVec_all = genotypeVec_all[xvals_all>0]
                
            # assign values to bins
            if auto_xBin == True:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            else:
                xBinEdges = xBinEdges_row[condIndx]
                if len(xBinEdges) == 1:
                    xBinEdges = xBinEdges[0]
                numxbins = len(xBinEdges)-1
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            
            bin_indices = np.digitize(xvals_all, xBinEdges)
            data = []
            xnames = []
            for kk in range(numxbins):
                # if multipleGenotypes == True:
                #     genotype_xbin = genotypeVec_all[bin_indices==(kk+1)]
                qVec_xbin = qVec_all[bin_indices==(kk+1)]
                data.append(qVec_xbin)
                if xlabeltype[rowIndx] == 'binmid':
                    xnames.append(str(round(xBinMid[kk],rounding[rowIndx]))) 
                elif xlabeltype[rowIndx] == 'binrange':
                    xnames.append(str(round(xBinEdges[kk],rounding[rowIndx])) + '-' + str(round(xBinEdges[kk+1],rounding[rowIndx]))) 
            if ifcalstats[rowIndx] == True: # calculate statistical significance 
                # between neighboring distributions
                pvalVec = []
                for kk in range(numxbins-1):
                    t_stat, p_val = stats.ttest_ind(data[kk], data[kk+1], equal_var=False)
                    pvalVec.append(p_val)
                ifsigVec = (np.array(pvalVec) <= alpha)
                # between first and last bin
                if numxbins > 2:
                    t_stat, p_val = stats.ttest_ind(data[0], data[numxbins-1], equal_var=False)
                    ifsig_firstlast = (p_val <= alpha)
                
            if ifshowNsamples[rowIndx] == True: # show number of data samples in each bin
                nsamplesVec = []
                for kk in range(numxbins):
                    nsamplesVec.append(len(data[kk]))
            
            if (numrows > 1) & (numconds > 1):
                axcurr = ax[rowIndx,condIndx]
            elif numrows == 1:
                axcurr = ax[condIndx]
            elif numconds == 1:
                axcurr = ax[rowIndx]
                
            axcurr.boxplot(data)
            axcurr.set_xticklabels(xnames)
            axcurr.set_ylabel(yname)
            if iflogx[rowIndx] == False:
                axcurr.set_title(xVar + '(' + cond + ')')
            else:
                axcurr.set_title('log10(' + xVar + ') (' + cond + ')')
            # statistical annotation
            if ifcalstats[rowIndx] == True:
                for kk in range(numxbins-1):
                    if ifsigVec[kk] == True:
                        x1, x2 = kk+1, kk+2
                        y, h, col = np.max(qVec_all) + 0.1, 0.15, 'k'
                        axcurr.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c=col)
                        axcurr.text((x1+x2)*.5, y+h, "s", ha='center', va='bottom', color=col)
                if (numxbins > 2) & (ifsig_firstlast == True):
                    y, h, col = np.max(qVec_all) + 0.3, 0.15, 'k'
                    axcurr.plot([1, 1, numxbins, numxbins], [y, y+h, y+h, y], lw=1.0, c=col)
                    axcurr.text((1+numxbins)*.5, y+h, "s", ha='center', va='bottom', color=col)
            # number of data samples
            if ifshowNsamples[rowIndx] == True:
                for kk in range(numxbins):
                    axcurr.text(kk+1, np.median(data[kk]) + 0.03, 
                                              'n = ' + str(nsamplesVec[kk]), ha='center', 
                                              va='bottom', color='k')
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)
                        
                                
# function for plotting boxplot showing the distribution of a specified quantity of interest as a function of 
# another specified variable across all flies for different experimental conditions.
# Unlike the function 'plot_boxplot_xcontinuous', here the x variable is fixed, but we allow different selection 
# criteria for the data points (which will be in different columns), while the different conditions will be in 
# separate rows.    
def plot_boxplot_varycriteria(df,qOI,xVar,xBinEdges_all,criteria,ifshowNsamples = False,ifcalstats = False,iflogx = False,
                             iflogq = False, alpha = 0.05,xlabeltype = 'binmid', rounding = 2, colwidth = 5, rowwidth = 5,
                             auto_xBin = False, autoNumxbins = 4):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    numcols = len(criteria)
    
    if iflogq == False:
        yname = qOI
    else:
        yname = 'log10(' + qOI + ')'
    if iflogx == False:
        xname = xVar
    else:
        xname = 'log10(' + xVar + ')'
    
    f,ax=plt.subplots(numconds,numcols,figsize=(numcols*colwidth,numconds*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        conds_df = df.loc[(df.condition == cond)]
        if auto_xBin == False:
            xBinEdges_cond = xBinEdges_all[condIndx]
        
        for keyIndx in range(numcols):
            
            rel_df = conds_df
            title = cond + '('
            for key in criteria[keyIndx].keys():
                minvalue = criteria[keyIndx][key]['min']
                maxvalue = criteria[keyIndx][key]['max']
                rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                title = title + (str(round(minvalue,-int(np.floor(np.log10(abs(minvalue)))))) + '<' + key + '<'
                        + str(round(maxvalue,-int(np.floor(np.log10(abs(maxvalue)))))) + ',')
            title = title + ')'
            
            qVec_all = rel_df[qOI].values
            xvals_all = rel_df[xVar].values
            if iflogq == True:
                xvals_all = xvals_all[qVec_all>0]
                qVec_all = np.log10(qVec_all[qVec_all>0])
            if iflogx == True:
                qVec_all = qVec_all[xvals_all > 0]
                xvals_all = np.log10(xvals_all[xvals_all>0])
            
            # assign values to bins
            if auto_xBin == False:
                xBinEdges = xBinEdges_cond[keyIndx]
                numxbins = len(xBinEdges)-1
            else:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            bin_indices = np.digitize(xvals_all, xBinEdges)
            data = []
            xnames = []
            for kk in range(numxbins):
                data.append(qVec_all[bin_indices==(kk+1)])
                if xlabeltype == 'binmid':
                    xnames.append(str(round(xBinMid[kk],rounding))) 
                elif xlabeltype == 'binrange':
                    xnames.append(str(round(xBinEdges[kk],rounding)) + '-' + str(round(xBinEdges[kk+1],rounding))) 
            if ifcalstats == True: # calculate statistical significance 
                # between neighboring distributions
                pvalVec = []
                for kk in range(numxbins-1):
                    t_stat, p_val = stats.ttest_ind(data[kk], data[kk+1], equal_var=False)
                    pvalVec.append(p_val)
                ifsigVec = (np.array(pvalVec) <= alpha)
                # between first and last bin
                if numxbins > 2:
                    t_stat, p_val = stats.ttest_ind(data[0], data[numxbins-1], equal_var=False)
                    ifsig_firstlast = (p_val <= alpha)
                
            if ifshowNsamples == True: # show number of data samples in each bin
                nsamplesVec = []
                for kk in range(numxbins):
                    nsamplesVec.append(len(data[kk]))
                
            if (numcols > 1) & (numconds > 1):
                ax[condIndx,keyIndx].boxplot(data)
                ax[condIndx,keyIndx].set_xticklabels(xnames)
                ax[condIndx,keyIndx].set_ylabel(yname)
                ax[condIndx,keyIndx].set_xlabel(xname)
                ax[condIndx,keyIndx].set_title(title)
                # statistical annotation
                if ifcalstats == True:
                    for kk in range(numxbins-1):
                        if ifsigVec[kk] == True:
                            x1, x2 = kk+1, kk+2
                            y, h, col = np.max(qVec_all) + 0.1, 0.15, 'k'
                            ax[condIndx,keyIndx].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c=col)
                            ax[condIndx,keyIndx].text((x1+x2)*.5, y+h, "s", ha='center', va='bottom', color=col)
                    if (numxbins > 2) & (ifsig_firstlast == True):
                        y, h, col = np.max(qVec_all) + 0.3, 0.15, 'k'
                        ax[condIndx,keyIndx].plot([1, 1, numxbins, numxbins], [y, y+h, y+h, y], lw=1.0, c=col)
                        ax[condIndx,keyIndx].text((1+numxbins)*.5, y+h, "s", ha='center', va='bottom', color=col)
                # number of data samples
                if ifshowNsamples == True:
                    for kk in range(numxbins):
                        ax[condIndx,keyIndx].text(kk+1, np.median(data[kk]) + 0.03, 'n = ' + str(nsamplesVec[kk]), 
                                                  ha='center', va='bottom', color='k')
                        
            elif numcols == 1:
                ax[condIndx].boxplot(data)
                ax[condIndx].set_xticklabels(xnames)
                ax[condIndx].set_ylabel(yname)
                ax[condIndx].set_xlabel(xname)
                ax[condIndx].set_title(title)
                # statistical annotation
                if ifcalstats == True:
                    for kk in range(numxbins-1):
                        if ifsigVec[kk] == True:
                            x1, x2 = kk+1, kk+2
                            y, h, col = np.max(qVec_all) + 0.1, 0.15, 'k'
                            ax[condIndx].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                            ax[condIndx].text((x1+x2)*.5, y+h, "s", ha='center', va='bottom', color=col)
                    if (numxbins > 2) & (ifsig_firstlast == True):
                        y, h, col = np.max(qVec_all) + 0.3, 0.15, 'k'
                        ax[condIndx].plot([1, 1, numxbins, numxbins], [y, y+h, y+h, y], lw=1.0, c=col)
                        ax[condIndx].text((1+numxbins)*.5, y+h, "s", ha='center', va='bottom', color=col)
                # number of data samples
                if ifshowNsamples == True:
                    for kk in range(numxbins):
                        ax[condIndx].text(kk+1, np.median(data[kk]) + 0.03, 'n = ' + str(nsamplesVec[kk]), 
                                          ha='center', va='bottom', color='k')
                
            elif numconds == 1:
                ax[keyIndx].boxplot(data)
                ax[keyIndx].set_xticklabels(xnames)
                ax[keyIndx].set_ylabel(yname)
                ax[keyIndx].set_xlabel(xname)
                ax[keyIndx].set_title(title)
                # statistical annotation
                if ifcalstats == True:
                    for kk in range(numxbins-1):
                        if ifsigVec[kk] == True:
                            x1, x2 = kk+1, kk+2
                            y, h, col = np.max(qVec_all) + 0.1, 0.15, 'k'
                            ax[keyIndx].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                            ax[keyIndx].text((x1+x2)*.5, y+h, "s", ha='center', va='bottom', color=col)
                    if (numxbins > 2) & (ifsig_firstlast == True):
                        y, h, col = np.max(qVec_all) + 0.3, 0.15, 'k'
                        ax[keyIndx].plot([1, 1, numxbins, numxbins], [y, y+h, y+h, y], lw=1.0, c=col)
                        ax[keyIndx].text((1+numxbins)*.5, y+h, "s", ha='center', va='bottom', color=col)
                # number of data samples
                if ifshowNsamples == True:
                    for kk in range(numxbins):
                        ax[keyIndx].text(kk+1, np.median(data[kk]) + 0.03, 'n = ' + str(nsamplesVec[kk]), 
                                         ha='center', va='bottom', color='k')
                
# function for plotting distribution of a quantity of interest across all flies 
# in the same condition for different criteria
def plot_distribution_varycriteria(df,qOI,names_cond, conds2include, criteria, numbins = 100, alpha = 0.5,
                      iflog = False, ifpdf = True, fitType = None, colwidth = 5, rowwidth = 5, 
                      ifoverlap = False, otherparams = None, fn2save = None):
    numrows = len(conds2include)
    if ifoverlap == False:
        numcols = len(criteria)
    else:
        numcols = 1
        
    if iflog == False:
        xname = qOI
    else:
        xname = "log10(" + qOI +")"
    if fitType != None:
        numcols = numcols + 2
    fig, ax = plt.subplots(numrows,numcols, figsize=(numcols*colwidth, numrows*rowwidth))
    for kk in range(numrows):
        conditions = names_cond[[value for value in conds2include.values()][kk]]
        conds_df = df.loc[np.isin(df.condition,conditions)]
        
        titlename_base = ''
        for condIndx in range(len(conditions)):
            titlename_base = titlename_base + conditions[condIndx] + ','
        
        if ifoverlap == True:
            legendname = []
            qvals_all = conds_df[qOI].values
            if iflog == True:
                qvals_all = np.log10(qvals_all[qvals_all>0])
            else:
                qvals_all = qvals_all[~np.isnan(qvals_all)]
            binEdges = np.histogram_bin_edges(qvals_all,numbins);
        
        for keyIndx in range(len(criteria)):
            rel_df = conds_df
            criteriaName_curr = ''
            for key in criteria[keyIndx].keys():
                if criteriaName_curr != '':
                    criteriaName_curr = criteriaName_curr + ','
                minvalue = criteria[keyIndx][key]['min']
                maxvalue = criteria[keyIndx][key]['max']
                rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                criteriaName_curr = criteriaName_curr + str(round(minvalue,2)) + '$\leq$' + key + '$\leq$' + str(round(maxvalue,2))  
            
            qoIvec = rel_df[qOI].values
            qoIvec = qoIvec[~np.isnan(qoIvec)]
            
            if iflog == False:
                data = qoIvec
            else:
                data = np.log10(qoIvec[qoIvec>0])
                
            if fitType == 'Normal':
                mu, sigma = stats.norm.fit(data)
                xscan = np.linspace(np.min(data),np.max(data),100) 
                best_fit_line = stats.norm.pdf(xscan, mu, sigma)
                titlename = titlename_base + 'mu = ' + str(round(mu,2)) + ', sd = ' + str(round(sigma,2))
                
            elif fitType == 'GMM': # gaussian mixture model
                Nmin = otherparams['Nmin']
                Nmax = otherparams['Nmax']
#                 (weights, means, covariances) = fitGMM(data,Nmin,Nmax)
                M_best = fitGMM(data.reshape(-1,1),Nmin,Nmax)
                Nopt = len(M_best.weights_)
        
                xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 1000)
                logprob = M_best.score_samples(xscan.reshape(-1, 1))
                responsibilities = M_best.predict_proba(xscan.reshape(-1, 1))
                best_fit_line = np.exp(logprob)
                pdf_individual = responsibilities * best_fit_line[:, np.newaxis]
                
                titlename = titlename_base
            else:
                titlename = titlename_base
            if ifoverlap == False:
                titlename = titlename + '(' + criteriaName_curr + ')'
        
            if (numrows > 1):
                if (ifoverlap == False):
                    axhist = ax[kk,keyIndx]
                else:
                    if numcols > 1:
                        axhist = ax[kk,1]
                    else:
                        axhist = ax[kk]
                if (fitType != None):
                    ax_mu = ax[kk,numcols-2]
                    ax_sigma = ax[kk,numcols-1]
            else:
                if (numcols == 1):
                    axhist = ax
                elif (ifoverlap == False):
                    axhist = ax[keyIndx]
                else:
                    axhist = ax[kk,1]
                if (fitType != None):
                    ax_mu = ax[numcols-2]
                    ax_sigma = ax[numcols-1]
           
            if ifoverlap == False:
                axhist.hist(data,numbins, density = ifpdf,alpha = alpha)   
            else:
                axhist.hist(data,binEdges, density = ifpdf,alpha = alpha)   
                
            if ifoverlap == True:
                legendname.append(criteriaName_curr)
        
            if fitType != None:
                axhist.plot(xscan, best_fit_line,'k',linewidth = 1)
                if ifoverlap == True:
                    legendname.append(criteriaName_curr + ', best fit')
                ax_mu.set_xlabel('which range')
                ax_sigma.set_xlabel('which range')
                ax_mu.set_ylabel('mu')
                ax_sigma.set_ylabel('sigma')
                if fitType == 'Normal':
                    ax_mu.scatter(keyIndx+1,mu,c='blue')
                    ax_sigma.scatter(keyIndx+1,sigma,c='blue')
                elif fitType == 'GMM':
                    axhist.plot(xscan, pdf_individual, '--k')
                    ax_mu.scatter(np.ones(Nopt)*(keyIndx+1),M_best.means_,
                                             s = M_best.weights_*100, c=list(range(1,Nopt+1)),cmap = 'Paired')
                    ax_sigma.scatter(np.ones(Nopt)*(keyIndx+1),M_best.covariances_,
                                             s = M_best.weights_*100, c=list(range(1,Nopt+1)),cmap = 'Paired')
            axhist.set_xlabel(xname)
            if (ifoverlap == True) & (ifpdf == False):
                axhist.set_ylabel("count")
            else:
                axhist.set_ylabel("P")
            axhist.set_title(titlename)
                
        if ifoverlap == True:
            axhist.legend(legendname)
            
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

# function for plotting distribution of a quantity of interest across all flies 
# in the same condition for different criteria
# In this v2, we allow fitting to specified distribution.
# Note that here we simplified things to only allow one set of conditions of interest
def plot_distribution_varycriteria_v2(df,qOI,conds2include, criteria, numbins = 100, alpha = 0.5, ifsubsample = False,
                      iflog = False, ifpdf = True, fitdist = None, fitparamsFuncType = None, colwidth = 5, rowwidth = 5, 
                      ifoverlap = False, iflogkey = False, otherparams = None, maxnumeval = 5000, fn2save = None):

    numcriteria = len(criteria)
    if ifoverlap == False:
        numcols = numcriteria
    else:
        numcols = 1

    if fitdist != None:
        # convert fitdist to a distribution object if relevant
        if isinstance(fitdist, str):
            fitdist = getattr(stats, fitdist)
    
        # extract parameter names of fitted distribution if relevant
        parameterNames = list_parameters(fitdist)
        numparams = len(parameterNames)

        # initialize list for storing fitted parameters
        fittedparams_all = [[0] * numparams] * numcriteria

        # middle of criteria ranges
        midrangeVec = np.zeros((numcriteria,1))
        
    if iflog == False:
        xname = qOI
    else:
        xname = "log10(" + qOI +")"

    # Extract criteria names
    criteriaNames_all = [0]*numcriteria
    for keyIndx in range(numcriteria):
        criteriaName_curr = ''
        resetname = True
        for key in criteria[keyIndx].keys():
            # if criteriaName_curr != '':
            #     criteriaName_curr = criteriaName_curr + ','
            if resetname == False:
                criteriaName_curr = criteriaName_curr + ','
            resetname = False
            minvalue = criteria[keyIndx][key]['min']
            maxvalue = criteria[keyIndx][key]['max']
            criteriaName_curr = criteriaName_curr + str(round(minvalue,2)) + '$\leq$' + key + '$\leq$' + str(round(maxvalue,2)) 
        criteriaNames_all[keyIndx] = criteriaName_curr
        if fitdist != None:
            midrangeVec[keyIndx] = (minvalue + maxvalue)/2
    
    fig, ax = plt.subplots(1,numcols, figsize=(numcols*colwidth, rowwidth))
    conds_df = df.loc[np.isin(df.condition,conds2include)]
    
    titlename = ''
    for condIndx in range(len(conds2include)):
        titlename = titlename + conds2include[condIndx] + ','
    
    if ifoverlap == True:
        legendname = []
        qvals_all = conds_df[qOI].values
        if iflog == True:
            qvals_all = np.log10(qvals_all[qvals_all>0])
        else:
            qvals_all = qvals_all[~np.isnan(qvals_all)]
        binEdges = np.histogram_bin_edges(qvals_all,numbins);
    
    for keyIndx in range(numcriteria):
        criteriaName_curr = criteriaNames_all[keyIndx]
        rel_df = conds_df
        for key in criteria[keyIndx].keys():
            minvalue = criteria[keyIndx][key]['min']
            maxvalue = criteria[keyIndx][key]['max']
            rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]

        qoIvec = rel_df[qOI].values
        # whether to keep all data values
        if ifsubsample == True:
            pkeep = np.min(rel_df.numSegs.values)/rel_df.numSegs.values
            uvec = np.random.random(len(pkeep))
            qoIvec = qoIvec[uvec<pkeep]
        
        qoIvec = qoIvec[~np.isnan(qoIvec)]        
        if iflog == False:
            data = qoIvec
        else:
            data = np.log10(qoIvec[qoIvec>0])

        if fitdist != None:
            fittedparams = fitdist.fit(data)
            fittedparams_all[keyIndx] = fittedparams
            xscan = np.linspace(np.min(data),np.max(data),100) 
            best_fit_line = fitdist.pdf(xscan, *fittedparams)
            ymax = np.max(best_fit_line)
            paramslegend = ''
            for paramIndx in range(numparams):
                paramslegend = paramslegend + parameterNames[paramIndx] + ' = ' + str(round(fittedparams[paramIndx],3))
                if paramIndx < numparams-1:
                    paramslegend = paramslegend + ','
            
        if ifoverlap == False:
            titlename = titlename + '(' + criteriaName_curr + ')'
    
        if (numcols == 1):
            axhist = ax
        else:
            axhist = ax[keyIndx]
            
        if ifoverlap == False:
            axhist.hist(data,numbins, density = ifpdf,alpha = alpha)   
        else:
            axhist.hist(data,binEdges, density = ifpdf,alpha = alpha)   
            
        if ifoverlap == True:
            legendname.append(criteriaName_curr)
               
    
        if fitdist != None:
            if ifoverlap == True:
                axhist.plot(xscan, best_fit_line,linewidth = 1)
                legendname.append(criteriaName_curr + ', best fit')
            else:
                axhist.plot(xscan, best_fit_line,'k',linewidth = 1)
                axhist.text(np.min(xscan),ymax/2,paramslegend)
        
        axhist.set_xlabel(xname)
        if (ifoverlap == True) & (ifpdf == False):
            axhist.set_ylabel("count")
        else:
            axhist.set_ylabel("P")
        axhist.set_title(titlename)
            
    if ifoverlap == True:
        axhist.legend(legendname)

    # save figures if desired             
    if fn2save != None:
        if fn2save.endswith('.pdf'):
            fn2save = fn2save[:-4]
        fig.savefig(fn2save + '.pdf', bbox_inches='tight', dpi=300)
        

    # if fitted distributions, also visualize how the fitted parameters change with our criteria
    if fitdist != None:
        fittedparamsMat = np.array(fittedparams_all)
        fig2, ax2 = plt.subplots(1,numparams, figsize=(numparams*colwidth, rowwidth))
        for paramIndx in range(numparams):
            ydata = fittedparamsMat[:,paramIndx]
            if iflogkey == False:
                xVarVec = midrangeVec
                xlabel = key
            elif iflogkey == True:
                xVarVec = np.log10(midrangeVec)
                xlabel = 'log10(' + key + ')'
            xdata = np.reshape(xVarVec,(len(xVarVec),))
            ax2[paramIndx].scatter(xdata,ydata)
            if fitparamsFuncType != None:
                if (fitparamsFuncType == 'PLfunc') or (fitparamsFuncType == 'ExpFunc'):
                    if ydata[1] < ydata[0]:
                        fitFuncType = 'decreasing' + fitparamsFuncType
                    else:
                        fitFuncType = 'increasing' + fitparamsFuncType
                else:
                    fitFuncType = fitparamsFuncType
                # extract initial guess and bounds for function parameters:
                param_guess, bounds = GetParamGuessAndBounds(xdata, ydata, fitFuncType)
                parameters, covariance = curve_fit(eval(fitFuncType), xdata, ydata, param_guess, bounds = bounds, maxfev=maxnumeval)
                # (parameters, covariance, infodict, mesg, ier) = curve_fit(eval(fitFuncType), xdata, ydata, param_guess, full_output=True)
                # print(infodict['fvec'])
                xscan = np.linspace(np.min(xdata),np.max(xdata),1000) 
                ax2[paramIndx].plot(xscan, eval(fitFuncType)(xscan,*parameters), '-', label='fit')
                # Get equation name
                EqnName = GetEqnString(fitFuncType,parameters)
                ax2[paramIndx].text((np.min(xscan)+np.max(xscan))/2,(np.min(ydata)+np.max(ydata))/2,EqnName,fontsize ='medium')

            ax2[paramIndx].set_xlabel(xlabel)
            ax2[paramIndx].set_ylabel(parameterNames[paramIndx])
            ax2[paramIndx].set_title(parameterNames[paramIndx])

        # save figures if desired             
        if fn2save != None:
            if fn2save.endswith('.pdf'):
                fn2save = fn2save[:-4]
            fig2.savefig(fn2save + '_fittedparams.pdf', bbox_inches='tight', dpi=300)


        return (fittedparamsMat, midrangeVec)
        
    
# function for comparing distribution of a quantity of interest for different genotypes
def plot_distribution_compareGenotypes(df_all, qOI, condnames_all, genotypenames, numbins = 100, ifabs = False,
                                       iflog = False, ifpdf = True, ifcdf = False, fitType = None, colwidth = 5, rowwidth = 5, 
                                       otherparams = None, ifoverlap = False, alpha = 0.5, histtype = 'bar', 
                                       ifcalstats = False, stattest = "Mann-Whitney", fn2save = None):
    
    numGenotypes = len(df_all)
    numconds = len(condnames_all)
    if ifabs == False:
        xname = qOI
    else:
        xname = '|' + qOI + '|'
    if iflog == True:
        xname = "log10(" + xname +")"
    
    if ifoverlap == False:
        numrows = numGenotypes
    elif ifoverlap == True:
        numrows = 1
        
    if ifcdf == False:
        yname = 'P'
    else:
        yname = 'C'
    
    fig, ax = plt.subplots(numrows,numconds, figsize=(numconds*colwidth, numrows*rowwidth))
    for condIndx in range(numconds):
        cond = condnames_all[condIndx]
        
        if ifoverlap == True:
            legendname = []
        
        nulldata = []
        for genotypeIndx in range(numGenotypes):
            genotype_df = df_all[genotypeIndx]
            rel_df = genotype_df.loc[genotype_df.condition == cond]
            
            if len(rel_df)>0:
                qoIvec = rel_df[qOI].values
                qoIvec = qoIvec[~np.isnan(qoIvec)]
                
                if ifabs == False:
                    data = qoIvec
                else:
                    data = np.abs(qoIvec)
                
                if iflog == True:
                    data = np.log10(data[data>0])
                    
                if ifcalstats == True:
                    if genotypeIndx == 0:
                        nulldata = data
                        statsInfo = ''
                    else:
                        if stattest == "Mann-Whitney":
                            _, pval = mannwhitneyu(data, nulldata) # method can be 'auto' or 'exact' or 'asymptotic'
                            # _, pval = mannwhitneyu(data, nulldata, method="exact") # method can be 'auto' or 'exact' or 'asymptotic'
                        elif stattest =="ttest_unequalvar":
                            _, p_val = stats.ttest_ind(data, nulldata, equal_var=False)
                        # statsInfo = ', p = ' + str(roundToSigFig(pval,2))
                        statsInfo = ', p = ' + str(round(pval, 2-int(np.floor(np.log10(abs(pval))))-1))
                        
                else:
                    statsInfo = ''
                    
                if fitType == 'Normal':
                    mu, sigma = stats.norm.fit(data)
                    xscan = np.linspace(np.min(data),np.max(data),100) 
                    if ifcdf == False:
                        best_fit_line = stats.norm.pdf(xscan, mu, sigma)
                    else:
                        best_fit_line = stats.norm.cdf(xscan, mu, sigma)
                    infoOI = 'mu = ' + str(round(mu,2)) + ', sd = ' + str(round(sigma,2))
                    
                elif fitType == 'GMM': # gaussian mixture model
                    Nmin = otherparams['Nmin']
                    Nmax = otherparams['Nmax']
                    M_best = fitGMM(data.reshape(-1,1),Nmin,Nmax)
                    Nopt = len(M_best.weights_)
            
                    xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 1000)
                    logprob = M_best.score_samples(xscan.reshape(-1, 1))
                    responsibilities = M_best.predict_proba(xscan.reshape(-1, 1))
                    if ifcdf == False:
                        best_fit_line = np.exp(logprob)
                    else:
                        means_all = []
                        weights_all = []
                        sigmas_all = []
                        for kk in range(Nopt):
                            means_all.append(M_best.means_[kk][0])
                            weights_all.append(M_best.weights_[kk])
                            sigmas_all.append(np.sqrt(M_best.covariances_[kk][0][0]))
                        best_fit_line = mix_norm_cdf(xscan, weights_all, means_all, sigmas_all)
                        
                    pdf_individual = responsibilities * best_fit_line[:, np.newaxis]
                    
                    if Nopt == 1:
                        infoOI = 'mu = ' + str(round(M_best.means_[0][0],2)) + ', sd = ' + str(round(np.sqrt(M_best.covariances_[0][0][0]),2))
                    else:
                        infoOI = ''
                        for kk in range(Nopt):
                            infoOI = infoOI + ('mu = ' + str(round(M_best.means_[kk][0],2)) + 
                                                ', sd = ' + str(round(np.sqrt(M_best.covariances_[kk][0][0]),2)) +
                                                ', w = ' + str(round(M_best.weights_[kk],2)) + '\n') 
                    
                if (numrows > 1) & (numconds > 1):
                    axcurr = ax[genotypeIndx,condIndx]
                    title = genotypenames[genotypeIndx] + ', ' + cond
                elif (numrows == 1):
                    axcurr = ax[condIndx]
                    title = cond
                elif (numconds == 1):
                    axcurr = ax[genotypeIndx]
                    title = genotypenames[genotypeIndx]
                    
                nvec, binvec, patches = axcurr.hist(data,numbins, density = ifpdf, cumulative = ifcdf, alpha = alpha, histtype = histtype,
                                                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][genotypeIndx])   
                if fitType != None:
                    axcurr.plot(xscan, best_fit_line,color = plt.rcParams['axes.prop_cycle'].by_key()['color'][genotypeIndx],linewidth = 1)
                    
                if (fitType == 'GMM') & (ifoverlap == False) & (ifcdf == False):
                    axcurr.plot(xscan, pdf_individual, '--k')
               
                if ifoverlap == True:
                    legendname.append(genotypenames[genotypeIndx] + statsInfo)
                else:
                    axcurr.text(np.min(data),np.max(nvec)-0.2,infoOI + statsInfo)
                    axcurr.set_xlabel(xname)
                    axcurr.set_ylabel(yname)
                    axcurr.set_title(title)
                    
        if ifoverlap == True:
            ax[condIndx].set_xlabel(xname)
            ax[condIndx].set_ylabel(yname)
            ax[condIndx].set_title(cond)
            ax[condIndx].legend(legendname)
              
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

# # function for overlaying multiple distributions of the same quantity but for different values of another quantity.
# # this gives the option of preserving the relative weights of the different distributions
# def overlay_distributions(df,qOI,)
            
# function for plotting distribution of a quantity of interest across all flies 
# either across conditions or over all conditions
def plot_distribution(df, qOI, names_cond, conds2include, criteria, numbins = 100, colwidth = 5, rowwidth = 5,
                      axislog = False, ifpdf = True, fitData = None, fitlogNormal = False, fn2save = None):
    numrows = len(conds2include)
    fig, ax = plt.subplots(numrows,4, figsize=(4*colwidth, numrows*rowwidth))
    
    for kk in range(numrows):
        conditions = names_cond[[value for value in conds2include.values()][kk]]
        rel_df = df.loc[np.isin(df.condition,conditions)]
        
        # include other relevant selection criteria
        for key in criteria.keys():
            minvalue = criteria[key]['min']
            maxvalue = criteria[key]['max']
            rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
        
        qoIvec = rel_df[qOI].values
        qoIvec = qoIvec[~np.isnan(qoIvec)]
        hist_qOI, bin_edges = np.histogram(qoIvec,numbins)
        binmid = (bin_edges[1:]+bin_edges[:-1])/2.
        logdata = np.log10(np.abs(qoIvec[qoIvec!=0]))
        if fitlogNormal == True:
            mu, sigma = stats.norm.fit(logdata)
            xscan = np.linspace(np.min(logdata),np.max(logdata),100) 
            best_fit_line_log = stats.norm.pdf(xscan, mu, sigma)
        if fitData == 'beta':
            a, b, loc, scale = stats.beta.fit(qoIvec[qoIvec>0], floc=0, fscale=1)
            uscan = np.linspace(0,1,100)
            best_fit_line = stats.beta.pdf(uscan,a,b)
            
        
        if numrows > 1:
            ax[kk,0].hist(qoIvec,numbins, log = axislog, density = ifpdf)   
            if fitData != None:
                ax[kk,0].plot(uscan,best_fit_line,'k',linewidth = 1)
            ax[kk,1].plot(np.log10(binmid),np.log10(hist_qOI),'x-')
            ax[kk,2].plot(binmid,np.log10(hist_qOI),'x-')
            ax[kk,3].hist(logdata,numbins, density = True)  
            if fitlogNormal == True:
                ax[kk,3].plot(xscan, best_fit_line_log,'k',linewidth = 1)
    
        else:
            ax[0].hist(qoIvec,numbins,log = axislog, density = ifpdf)   
            if fitData != None:
                ax[0].plot(uscan,best_fit_line,'k',linewidth = 1)
            ax[1].plot(np.log10(binmid),np.log10(hist_qOI),'x-')
            ax[2].plot(binmid,np.log10(hist_qOI),'x-')
            ax[3].hist(logdata,numbins, density = True) 
            if fitlogNormal == True:
                ax[kk,3].plot(xscan, best_fit_line_log,'k',linewidth = 1)
    
            
        titlename = ''
        for condIndx in range(len(conditions)):
            titlename = titlename + conditions[condIndx] + ','

        if numrows > 1:
            ax[kk,0].set_xlabel(qOI)
            ax[kk,0].set_ylabel("P")
            if fitData == None:
                ax[kk,0].set_title(titlename)
            elif fitData == 'beta':
                ax[kk,0].set_title(titlename + 'a = ' + str(round(a,2)) + ', b = ' + str(round(b,2)))
            ax[kk,1].set_xlabel("log10(" + qOI +")")
            ax[kk,1].set_ylabel("log10(P)")
            ax[kk,1].set_title(titlename)
            ax[kk,2].set_xlabel(qOI)
            ax[kk,2].set_ylabel("log10(P)")
            ax[kk,2].set_title(titlename)
            ax[kk,3].set_xlabel("log10(" + qOI +")")
            ax[kk,3].set_ylabel("P")
            if fitlogNormal == False:
                ax[kk,3].set_title(titlename)
            else:
                ax[kk,3].set_title(titlename + 'mu = ' + str(round(mu,2)) + ', sd = ' + str(round(sigma,2)))
        else:
            ax[0].set_xlabel(qOI)
            ax[0].set_ylabel("P")
            if fitData == None:
                ax[0].set_title(titlename)
            elif fitData == 'beta':
                ax[0].set_title(titlename + 'a = ' + str(round(a,2)) + ', b = ' + str(round(b,2)))
            ax[1].set_xlabel("log10(" + qOI +")")
            ax[1].set_ylabel("log10(P)")
            ax[1].set_title(titlename)
            ax[2].set_xlabel(qOI)
            ax[2].set_ylabel("log10(P)")
            ax[2].set_title(titlename)
            ax[3].set_xlabel("log10(" + qOI +")")
            ax[3].set_ylabel("P")
            if fitlogNormal == False:
                ax[3].set_title(titlename)
            else:
                ax[3].set_title(titlename + 'mu = ' + str(round(mu,2)) + ', sd = ' + str(round(sigma,2)))
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

# function for plotting distribution of a quantity of interest across all flies 
# either across conditions or over all conditions
# In this v2, we allow fitting to arbitrary specified distribution, and specify the fitted distribution over the data
def plot_distribution_v2(df, qOI, conds2include_dict, criteria, numbins = 100, colwidth = 5, rowwidth = 5,
                      iflog = False, ifpdf = True, ifcdf = False, fitdist = None, fn2save = None):
    numcols = len(conds2include_dict)
    fig, ax = plt.subplots(1, numcols, figsize=(numcols*colwidth, rowwidth))

    if fitdist != None:
        # convert fitdist to a distribution object if relevant
        if isinstance(fitdist, str):
            fitdist = getattr(stats, fitdist)
    
        # extract parameter names of fitted distribution if relevant
        parameterNames = list_parameters(fitdist)
        numparams = len(parameterNames)

    if iflog == False:
        qName = qOI
    else:
        qName = 'log10(' + qOI + ')'
    
    for kk in range(numcols):
        conditions = conds2include_dict[str(kk)]
        rel_df = df.loc[np.isin(df.condition,conditions)]
        
        # include other relevant selection criteria
        for key in criteria.keys():
            minvalue = criteria[key]['min']
            maxvalue = criteria[key]['max']
            rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
        
        qoIvec = rel_df[qOI].values
        qoIvec = qoIvec[~np.isnan(qoIvec)]
        if iflog == False:
            datavec = qoIvec
        elif iflog == True:
            datavec = np.log10(np.abs(qoIvec[qoIvec!=0]))
            
        if fitdist != None:
            # dist = getattr(stats, fitdist)
            fittedparams = fitdist.fit(datavec)
            xscan = np.linspace(np.min(datavec),np.max(datavec),100) 
            if ifcdf == False:
                best_fit_line = fitdist.pdf(xscan, *fittedparams)
            else:
                best_fit_line = fitdist.cdf(xscan, *fittedparams)
            paramslegend = ''
            for paramIndx in range(numparams):
                paramslegend = paramslegend + parameterNames[paramIndx] + ' = ' + str(round(fittedparams[paramIndx],3))
                if paramIndx < numparams-1:
                    paramslegend = paramslegend + ','
            
        titlename = ''
        for condIndx in range(len(conditions)):
            titlename = titlename + conditions[condIndx] + ','

        if numcols > 1:
            axcurr = ax[kk]
        else:
            axcurr = ax

        if ifcdf == False:
            axcurr.hist(datavec,numbins, density = ifpdf)   
        else:
            axcurr.hist(datavec,numbins, density = True, histtype = "step", cumulative = True)   
        if fitdist != None:
            axcurr.plot(xscan,best_fit_line,'k',linewidth = 1)
            ymax = np.max(best_fit_line)
            # axcurr.legend(legend)
            axcurr.text(np.min(xscan),ymax/2,paramslegend)
                
        axcurr.set_xlabel(qName)
        if ifcdf == False:
            axcurr.set_ylabel("P")
        else:
            axcurr.set_ylabel("C")
        axcurr.set_title(titlename)
        
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

    if fitdist != None:
        return fittedparams

# function for plotting distribution of a quantity of interest for individual flies 
# for a specified condition
def plot_distribution_eachfly(df, qOI, maxnumflies, criteria, numbins = 100, 
                              ifabs = False, iflog = False, colwidth = 5, rowwidth = 5,
                              outlierdetection = None, multipleGenotypes = False):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    
    if ifabs == False:
        xname = qOI
    else:
        xname = '|' + qOI + '|'
    if iflog == True:
        xname = "log10(" + xname +")"
    
    f,ax=plt.subplots(maxnumflies,numconds,figsize=(numconds*colwidth,maxnumflies*rowwidth))

    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        
        # include other relevant selection criteria
        for key in criteria.keys():
            minvalue = criteria[key]['min']
            maxvalue = criteria[key]['max']
            perCond_df = perCond_df.loc[(perCond_df[key] >= minvalue) & 
                                        (perCond_df[key] <= maxvalue)]
    
        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]
            
            title = cond + '(fly' + str(flyIndx+1) +')'
            if multipleGenotypes == True:
                gtype = rel_df.genotype.values[0]
                title = gtype + ',' + title
            
            qoIvec = rel_df[qOI].values
            if ifabs == True:
                qoIvec = np.abs(qoIvec)
            qoIvec = qoIvec[~np.isnan(qoIvec)]
            if iflog == True:
                qoIvec = np.log10(qoIvec[qoIvec>0])
                
#             hist_qOI, bin_edges = np.histogram(qoIvec,numbins)
#             binmid = (bin_edges[1:]+bin_edges[:-1])/2.
            
            if len(qoIvec) > 1: 
                if outlierdetection == 'medcouple':
                    M_fly = medcouple(qoIvec)
                    Q3, Q1 = np.percentile(qoIvec, [75 ,25])
                    IQR = Q3 - Q1
                    thres = Q3 + np.exp(3*M_fly)*1.5*IQR 
                    title = title + ', thres = ' + str(np.round(thres,2))
            if len(rel_df) > 0:
                ax[flyIndx,condIndx].hist(qoIvec,numbins)
                if outlierdetection != None:
                    ax[flyIndx,condIndx].axvline(thres,color = 'k')
                ax[flyIndx,condIndx].set_xlabel(xname)
                ax[flyIndx,condIndx].set_ylabel("P")
                ax[flyIndx,condIndx].set_title(title)
        
# function for plotting scatter plot between pairs of variables across all flies within the same condition
def plot_correlations(df, xVars, yVars, colInds, criteria, iflogx, iflogy, ifabsx, ifabsy, 
                      condcolors = {'0-125M_24hr': 'orangered', '0-125M_40hr': 'brown','0M_24hr': 'lightseagreen','0M_40hr':'teal'},
                      fn2save = None):
    names_cond = df.condition.unique()
    
    numcols = np.max(colInds) + 1
    numcomparisons = len(xVars)
    fig, axes = plt.subplots(numcomparisons,numcols, figsize=(numcols*5, numcomparisons*5))
    for comparisonIndx in range(numcomparisons):
        xVar = xVars[comparisonIndx]
        yVar = yVars[comparisonIndx]
        
        xname = xVar
        if ifabsx[comparisonIndx] == True:
            xname = '|' + xname + '|'
        if iflogx[comparisonIndx] == True:
            xname = 'log10(' + xname + ')'
        yname = yVar
        if ifabsy[comparisonIndx] == True:
            yname = '|' + yname + '|'
        if iflogy[comparisonIndx] == True:
            yname = 'log10(' + yname + ')'
        
        for colIndx in range(numcols):
            legend = []
            condsOI = np.where(colInds == colIndx)[0] 
            
            if (numcomparisons>1) & (numcols>1):
                axis = axes[comparisonIndx,colIndx]
            elif (numcomparisons==1):
                axis = axes[colIndx]
            elif (numcols==1):
                axis = axes[comparisonIndx]

            
            for kk in range(len(condsOI)):
                condIndx = condsOI[kk]
                cond = names_cond[condIndx]
                rel_df = df.loc[df.condition == cond]

                # include other relevant selection criteria
                for key in criteria.keys():
                    minvalue = criteria[key]['min']
                    maxvalue = criteria[key]['max']
                    rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]

                xdata = rel_df[xVar]
                ydata = rel_df[yVar]
                if ifabsx[comparisonIndx] == True:
                    xdata = np.abs(xdata)
                if ifabsy[comparisonIndx] == True:
                    ydata = np.abs(ydata)
                    
                nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
                if iflogx[comparisonIndx] == False:
                    xVec = xdata[~nas]
                else:
                    xVec = np.log10(xdata[~nas])
                if iflogy[comparisonIndx] == False:   
                    yVec = ydata[~nas]
                else:
                    yVec = np.log10(ydata[~nas])
                    
                sns.regplot(x=xVec, y=yVec, ax = axis, color = condcolors[cond])
                corrcoeff = xVec.corr(yVec)
                legend.append(cond + ', r = '+str(round(corrcoeff,3)))
            
            axis.set_xlabel(xname)
            axis.set_ylabel(yname)
            axis.legend(legend)
            
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)


# function for plotting scatter plot between pairs of variables for individual flies 
def plot_correlations_eachfly(df, xVar, yVar, maxnumflies, criteria, 
                              iflogx = False, iflogy = False, ifabsx = False, ifabsy = False,
                              multipleGenotypes = False, ifoverlap = False,
                              colwidth = 5, rowwidth = 5, fn2save = None):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    
    xname = xVar
    if ifabsx == True:
        xname = '|' + xname + '|'
    if iflogx == True:
        xname = 'log10(' + xname + ')'
    yname = yVar
    if ifabsy == True:
        yname = '|' + yname + '|'
    if iflogy == True:
        yname = 'log10(' + yname + ')'
        
    if ifoverlap == False:
        fig, axes = plt.subplots(maxnumflies,numconds, figsize=(numconds*colwidth, maxnumflies*rowwidth))
    else:
        if multipleGenotypes == True:
            names_Genotypes = df.genotype.unique()
            numGenotypes = len(names_Genotypes)
        else:
            numGenotypes = 1
        fig, axes = plt.subplots(numGenotypes,numconds, figsize=(numconds*colwidth, numGenotypes*rowwidth))
        
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        
        # include other relevant selection criteria
        for key in criteria.keys():
            minvalue = criteria[key]['min']
            maxvalue = criteria[key]['max']
            perCond_df = perCond_df.loc[(perCond_df[key] >= minvalue) & 
                                        (perCond_df[key] <= maxvalue)]
    
        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]

            if len(rel_df)>1:
                xdata = rel_df[xVar]
                ydata = rel_df[yVar]
                if ifabsx == True:
                    xdata = np.abs(xdata)
                if ifabsy == True:
                    ydata = np.abs(ydata)
                nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
                if iflogx == False:
                    xVec = xdata[~nas]
                else:
                    xVec = np.log10(xdata[~nas])
                if iflogy == False:   
                    yVec = ydata[~nas]
                else:
                    yVec = np.log10(ydata[~nas])
                
                if ifoverlap == False:
                    axcurr = axes[flyIndx,condIndx]
                else:
                    if (multipleGenotypes == True) & (numGenotypes > 1):
                        gtype = rel_df.genotype.values[0]
                        whichgenotype = np.where(names_Genotypes == gtype)[0][0]
                        axcurr = axes[whichgenotype,condIndx]
                    else:
                        axcurr= axes[condIndx]
                
                if len(xVec) > 1:
                    sns.regplot(x=xVec, y=yVec, ax = axcurr)
                    corrcoeff = xVec.corr(yVec)
                    
                    axcurr.set_xlabel(xname)
                    axcurr.set_ylabel(yname)
                    
                    if multipleGenotypes == False:
                        title = cond + '(fly' + str(flyIndx+1) + ', r = '+str(round(corrcoeff,3)) + ')'
                    else:
                        gtype = rel_df.genotype.values[0]
                        title = gtype + ',' + cond + '(fly' + str(flyIndx+1) + ', r = '+str(round(corrcoeff,3)) + ')'
                    
                    axcurr.set_title(title)
            
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

# This function creates a scatter plot between pairs of variables across all flies within the same condition, 
# with the scatter points colored by a third variable (e.g. state variable of interest)
def createScatterPlot(df, xVars, yVars, indVars, criteria, iflogx, iflogy, iflogind,
                      ifabsx = False, ifabsy = False, ifabsind = False, ifcalcorr = False, 
                      maxmksize = 5, minmksize = 2, colwidth = 5, rowwidth = 5):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    
    numcomparisons = len(xVars)
    
    if ifabsx == False:
        ifabsx = [False]*numcomparisons
    if ifabsy == False:
        ifabsy = [False]*numcomparisons
    if ifabsind == False:
        ifabsind = [False]*numcomparisons
    if ifcalcorr == False:
        ifcalcorr = [False]*numcomparisons
        
    fig, axes = plt.subplots(numcomparisons,numconds, figsize=(numconds*colwidth, numcomparisons*rowwidth))
    for comparisonIndx in range(numcomparisons):
        xVar = xVars[comparisonIndx]
        yVar = yVars[comparisonIndx]
        indVar = indVars[comparisonIndx]
        
        xname = xVar
        if ifabsx[comparisonIndx] == True:
            xname = '|' + xname + '|'
        if iflogx[comparisonIndx] == True:
            xname = 'log10(' + xname + ')'
        yname = yVar
        if ifabsy[comparisonIndx] == True:
            yname = '|' + yname + '|'
        if iflogy[comparisonIndx] == True:
            yname = 'log10(' + yname + ')'
            
        # extract minimum and maximum ind values for coloring scatter points
        ind_allVals = df[indVar].values + 1e-4
        if ifabsind[comparisonIndx] == True:
            ind_allVals = np.abs(ind_allVals)
        if iflogind[comparisonIndx] == True:   
            ind_allVals = ind_allVals[ind_allVals>0]
            ind_allVals = np.log10(ind_allVals)
        colormin = np.min(ind_allVals)
        colormax = np.max(ind_allVals)
        
        for condIndx in range(numconds):
            legend = []
            cond = names_cond[condIndx]
            rel_df = df.loc[df.condition == cond]

            # include other relevant selection criteria
            for key in criteria[comparisonIndx].keys():
                minvalue = criteria[comparisonIndx][key]['min']
                maxvalue = criteria[comparisonIndx][key]['max']
                rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                
            xdata = rel_df[xVar]
            ydata = rel_df[yVar]
            inddata = rel_df[indVar]
            if ifabsx[comparisonIndx] == True:
                xdata = np.abs(xdata)
            if ifabsy[comparisonIndx] == True:
                ydata = np.abs(ydata)
            if ifabsind[comparisonIndx] == True:
                inddata = np.abs(inddata)

            nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
            if iflogx[comparisonIndx] == False:
                xVec = xdata[~nas]
            else:
                xVec = np.log10(xdata[~nas])
            if iflogy[comparisonIndx] == False:   
                yVec = ydata[~nas]
            else:
                yVec = np.log10(ydata[~nas])
            if iflogind[comparisonIndx] == False:   
                indVec = inddata[~nas]
            else:
                indVec = np.log10(inddata[~nas])
            indmin = np.min(indVec)
            indmax = np.max(indVec)
            sizeVec = minmksize + (indVec-indmin)/(indmax-indmin+1e-6)*(maxmksize-minmksize)

            # scatter plot
            if (numcomparisons > 1) & (numconds > 1):
                axcurr = axes[comparisonIndx,condIndx]
            elif (numcomparisons == 1):
                axcurr = axes[condIndx]
            elif (numconds == 1):
                axcurr = axes[comparisonIndx]
                
            axcurr.scatter(xVec, yVec, c = indVec, s = sizeVec, 
                                         vmin = colormin, vmax = colormax, cmap='rainbow')
            legend.append('data')
            # calculate correlation coefficient and draw linear regression line if desired
            if ifcalcorr[comparisonIndx] == True:
                # correlation coeff:
                corrcoeff = xVec.corr(yVec)
                # best fit line:
                b, a = np.polyfit(xVec, yVec, deg=1)
                xscan = np.linspace(np.min(xVec), np.max(xVec), 100)
                # Plot regression line:
                axcurr.plot(xscan, a + b * xscan, color="k", lw=2.5);
                legend.append('y = ' + str(round(a,3)) + '+' + str(round(b,3)) + 'x, r = ' + str(round(corrcoeff,3)))
                axes[comparisonIndx,condIndx].legend(legend)

            axcurr.set_xlabel(xname)
            axcurr.set_ylabel(yname)
            axcurr.set_title(cond)

# This function creates a scatter plot between pairs of variables across all flies within the same condition, 
# with the scatter points colored by a third variable (e.g. state variable of interest)
def createScatterPlot_compareGenotypes(df_all, names_cond, genotypenames, 
                                       xVars, yVars, indVars, criteria, iflogx, iflogy, iflogind,
                                       ifabsx = False, ifabsy = False, ifabsind = False, ifcalcorr = False, 
                                       maxmksize = 5, minmksize = 2, colwidth = 5, rowwidth = 5):
    numGenotypes = len(df_all)
    numconds = len(names_cond)
    numcomparisons = len(xVars)
    
    if ifabsx == False:
        ifabsx = [False]*numcomparisons
    if ifabsy == False:
        ifabsy = [False]*numcomparisons
    if ifabsind == False:
        ifabsind = [False]*numcomparisons
    if ifcalcorr == False:
        ifcalcorr = [False]*numcomparisons
        
    fig, axes = plt.subplots(numcomparisons,numconds, figsize=(numconds*colwidth, numcomparisons*rowwidth))
    for comparisonIndx in range(numcomparisons):
        xVar = xVars[comparisonIndx]
        yVar = yVars[comparisonIndx]
        indVar = indVars[comparisonIndx]
        
        xname = xVar
        if ifabsx[comparisonIndx] == True:
            xname = '|' + xname + '|'
        if iflogx[comparisonIndx] == True:
            xname = 'log10(' + xname + ')'
        yname = yVar
        if ifabsy[comparisonIndx] == True:
            yname = '|' + yname + '|'
        if iflogy[comparisonIndx] == True:
            yname = 'log10(' + yname + ')'
            
        for condIndx in range(numconds):
            legend = []
            cond = names_cond[condIndx]
            
            # current axis
            if (numcomparisons > 1) & (numconds > 1):
                axcurr = axes[comparisonIndx,condIndx]
            elif (numcomparisons == 1):
                axcurr = axes[condIndx]
            elif (numconds == 1):
                axcurr = axes[comparisonIndx]
                
            for genotypeIndx in range(numGenotypes):
                genotype_df = df_all[genotypeIndx]
                rel_df = genotype_df.loc[genotype_df.condition == cond]
                
                if len(rel_df)>0:
                    # include other relevant selection criteria
                    for key in criteria[comparisonIndx].keys():
                        minvalue = criteria[comparisonIndx][key]['min']
                        maxvalue = criteria[comparisonIndx][key]['max']
                        rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                        
                    xdata = rel_df[xVar]
                    ydata = rel_df[yVar]
                    inddata = rel_df[indVar]
                    if ifabsx[comparisonIndx] == True:
                        xdata = np.abs(xdata)
                    if ifabsy[comparisonIndx] == True:
                        ydata = np.abs(ydata)
                    if ifabsind[comparisonIndx] == True:
                        inddata = np.abs(inddata)
        
                    nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
                    if iflogx[comparisonIndx] == False:
                        xVec = xdata[~nas]
                    else:
                        xVec = np.log10(xdata[~nas])
                    if iflogy[comparisonIndx] == False:   
                        yVec = ydata[~nas]
                    else:
                        yVec = np.log10(ydata[~nas])
                    if iflogind[comparisonIndx] == False:   
                        indVec = inddata[~nas]
                    else:
                        indVec = np.log10(inddata[~nas])
                    indmin = np.min(indVec)
                    indmax = np.max(indVec)
                    sizeVec = minmksize + (indVec-indmin)/(indmax-indmin+1e-6)*(maxmksize-minmksize)
        
                    # scatter plot
                    axcurr.scatter(xVec, yVec, 
                                   c = plt.rcParams['axes.prop_cycle'].by_key()['color'][genotypeIndx], 
                                   s = sizeVec)
                    legend.append(genotypenames[genotypeIndx] + ' data')
                    # calculate correlation coefficient and draw linear regression line if desired
                    if ifcalcorr[comparisonIndx] == True:
                        # correlation coeff:
                        corrcoeff = xVec.corr(yVec)
                        # best fit line:
                        b, a = np.polyfit(xVec, yVec, deg=1)
                        xscan = np.linspace(np.min(xVec), np.max(xVec), 100)
                        # Plot regression line:
                        axcurr.plot(xscan, a + b * xscan, 
                                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][genotypeIndx], lw=2.5);
                        legend.append(genotypenames[genotypeIndx] +', y = ' + str(round(a,3)) + '+' + str(round(b,3)) + 'x, r = ' + str(round(corrcoeff,3)))
        
            axcurr.set_xlabel(xname)
            axcurr.set_ylabel(yname)
            axcurr.set_title(cond)
            axcurr.legend(legend)
            
# This function creates a scatter plot between a given pair of variables for each individual fly, 
# with the scatter points with size dependent on a third variable (e.g. state variable of interest)
def createScatterPlot_eachfly(df, xVar, yVar, indVar, maxnumflies, criteria, iflogx, iflogy, iflogind,
                      ifabsx = False, ifabsy = False, ifabsind = False, ifcalcorr = False, 
                      maxmksize = 5, minmksize = 2, minlinewidth = 1, maxlinewidth = 2, LWtype = 'Fixed', colwidth = 5, rowwidth = 5,
                      multipleGenotypes = False, ifoverlap = False, minnumsamples = 2):
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    cond_colors = np.random.rand(numconds,3)
    
    if multipleGenotypes == True:
        names_Genotypes = df.genotype.unique()
        numGenotypes = len(names_Genotypes)
    else:
        numGenotypes = 1
    
    xname = xVar
    if ifabsx == True:
        xname = '|' + xname + '|'
    if iflogx == True:
        xname = 'log10(' + xname + ')'
    yname = yVar
    if ifabsy == True:
        yname = '|' + yname + '|'
    if iflogy == True:
        yname = 'log10(' + yname + ')'
        
    # # extract minimum and maximum ind values for coloring scatter points
    # ind_allVals = df[indVar].values
    # if ifabsind == True:
    #     ind_allVals = np.abs(ind_allVals)
    # if iflogind == True:   
    #     ind_allVals = ind_allVals[ind_allVals>0]
    #     ind_allVals = np.log10(ind_allVals)
    # colormin = np.min(ind_allVals)
    # colormax = np.max(ind_allVals)
    
    # colormat = np.random.choice(range(255),size=[maxnumflies,3])
    
    # include desired selection criteria
    for key in criteria.keys():
        minvalue = criteria[key]['min']
        maxvalue = criteria[key]['max']
        df = df.loc[(df[key] >= minvalue) & (df[key] <= maxvalue)]
    
    
    # initialize figure and storage dictionaries for 
    # correlation coefficient, number of samples, gradient of best-fit-line  
    if ifoverlap == False:
        numrows = maxnumflies
    else:
        numrows = numGenotypes
    numcols = numconds
    if ifcalcorr == True:
        numcols = numcols + 3
        if numGenotypes > 1:
            numrows = numrows + 3
            keynames = [(x,y) for x in names_Genotypes for y in names_cond]
            genotype_colors = np.random.rand(numGenotypes,3)
        else:
            keynames = [y for y in names_cond]
        corrcoeff_dict = {key: [] for key in keynames}
        numsamples_dict = {key: [] for key in keynames}
        slope_dict = {key: [] for key in keynames}
        
    fig, axes = plt.subplots(numrows,numcols, figsize=(numcols*colwidth, numrows*rowwidth))
    
    for condIndx in range(numconds):
        legend = []
        cond = names_cond[condIndx]
        perCond_df = df.loc[(df.condition == cond)]
        
        if (multipleGenotypes == True) & (numGenotypes > 1):
            Genotypes_currcond = perCond_df.genotype.unique()
            numGenotypes_currcond = len(Genotypes_currcond)
        else:
            numGenotypes_currcond = 1

        for flyIndx in range(len(perCond_df.fly.unique())):
            fly = perCond_df.fly.unique()[flyIndx]
            rel_df = perCond_df.loc[perCond_df.fly==fly]

            if len(rel_df) >= minnumsamples:
                if (multipleGenotypes == True) & (numGenotypes > 1):
                    gtype = rel_df.genotype.values[0]
                    whichgenotype = np.where(names_Genotypes == gtype)[0][0]
                
                xdata = rel_df[xVar]
                ydata = rel_df[yVar]
                if ifabsx == True:
                    xdata = np.abs(xdata)
                if ifabsy == True:
                    ydata = np.abs(ydata)
                if indVar != 'numsamples':
                    inddata = rel_df[indVar]
                    if ifabsind == True:
                        inddata = np.abs(inddata)
        
                nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
                if iflogx == False:
                    xVec = xdata[~nas]
                else:
                    xVec = np.log10(xdata[~nas])
                if iflogy == False:   
                    yVec = ydata[~nas]
                else:
                    yVec = np.log10(ydata[~nas])
                if indVar != 'numsamples':
                    if iflogind == False:   
                        indVec = inddata[~nas]
                    else:
                        indVec = np.log10(inddata[~nas])
                    indmin = np.min(indVec)
                    indmax = np.max(indVec)
                    sizeVec = minmksize + (indVec-indmin)/(indmax-indmin+1e-6)*(maxmksize-minmksize)
                elif indVar == 'numsamples':
                    sizeVec = minmksize + len(xVec)
                
                if ifoverlap == False:
                    axcurr = axes[flyIndx,condIndx]
                else:
                    if (multipleGenotypes == True) & (numGenotypes > 1):
                        axcurr = axes[whichgenotype,condIndx]
                    else:
                        axcurr= axes[condIndx]
                
                if len(xVec) >= minnumsamples:
                    # scatter plot
                    # rndcolor = np.random.choice(range(255),size=3)
                    rndcolor = np.random.rand(1,3)
                    axcurr.scatter(xVec, yVec, c = np.tile(rndcolor,(len(xVec),1)), s = sizeVec)
                    # axcurr.scatter(xVec, yVec, c = indVec, s = sizeVec, 
                    #                          vmin = colormin, vmax = colormax, cmap='rainbow')
                    
                    if ifoverlap == False:
                        legend.append('data')
                    # calculate correlation coefficient and draw linear regression line if desired
                    if ifcalcorr == True:
                        # correlation coeff:
                        corrcoeff = xVec.corr(yVec)
                        # best fit line:
                        b, a = np.polyfit(xVec, yVec, deg=1)
                        xscan = np.linspace(np.min(xVec), np.max(xVec), 100)
                        # Plot regression line
                        if LWtype == 'Fixed':
                            LW = maxlinewidth
                        elif LWtype == 'R2': # goodness of fit
                            LW = minlinewidth + (maxlinewidth-minlinewidth)*corrcoeff**2
                        axcurr.plot(xscan, a + b * xscan, color=rndcolor.tolist()[0], lw=LW);
                        if ifoverlap == False:
                            legend.append('y = ' + str(round(a,3)) + str(round(b,3)) + 'x, r = ' + str(round(corrcoeff,3)))
                            axcurr.legend(legend)
                        # store data
                        if (multipleGenotypes == True) & (numGenotypes > 1):
                            keyOI = (gtype, cond)
                        else:
                            keyOI = cond
                        corrcoeff_dict[keyOI].append(corrcoeff)
                        numsamples_dict[keyOI].append(len(xVec))
                        slope_dict[keyOI].append(b)
                        
                    axcurr.set_xlabel(xname)
                    axcurr.set_ylabel(yname)
                    if multipleGenotypes == False:
                        title = cond 
                    else:
                        gtype = rel_df.genotype.values[0]
                        title = gtype + ',' + cond 
                    axcurr.set_title(title)
        
        # if plot all flies on the same plot, fit data from all flies
        if (ifoverlap == True) & (ifcalcorr == True):
            for gkk in range(numGenotypes_currcond):
                if (multipleGenotypes == True):
                    gtype = Genotypes_currcond[gkk]
                    whichgenotype = np.where(names_Genotypes == gtype)[0][0]
                    rel_df = perCond_df.loc[perCond_df.genotype==gtype]
                    axcurr = axes[whichgenotype,condIndx]
                else: 
                    rel_df = perCond_df
                    axcurr = axes[condIndx]
                xdata = rel_df[xVar]
                ydata = rel_df[yVar]
                if ifabsx == True:
                    xdata = np.abs(xdata)
                if ifabsy == True:
                    ydata = np.abs(ydata)
                nas = np.logical_or(np.logical_or(np.logical_or(np.isnan(xdata),np.isnan(ydata)),xdata==0),ydata==0)
                if iflogx == False:
                    xVec = xdata[~nas]
                else:
                    xVec = np.log10(xdata[~nas])
                if iflogy == False:   
                    yVec = ydata[~nas]
                else:
                    yVec = np.log10(ydata[~nas])
                corrcoeff = xVec.corr(yVec)
                # best fit line:
                b, a = np.polyfit(xVec, yVec, deg=1)
                xscan = np.linspace(np.min(xVec), np.max(xVec), 100)
                yscan = a + b * xscan
                axcurr.plot(xscan, yscan, color='k', lw=maxlinewidth*2);
                legend = 'y = ' + str(round(a,3)) + '+' + str(round(b,3)) + 'x, r = ' + str(round(corrcoeff,3))
                # axcurr.legend(legend)
                axcurr.text(np.min(xVec),(np.min(yVec) + np.min(yscan))/2,legend)
                # axcurr.text((x1+x2)*.5, y+h, "s", ha='center', va='bottom', color=col)
                
        # plot summary statistics comparing across genotypes (for the same condition)
        if (ifcalcorr == True) & (numGenotypes_currcond > 1):
            for gkk in range(numGenotypes_currcond):
                genotypeOI = Genotypes_currcond[gkk]
                keyOI = (genotypeOI,cond)
                corrcoeffVec = np.array(corrcoeff_dict[keyOI])
                numsamplesVec = np.array(numsamples_dict[keyOI])
                sVec = minmksize + numsamplesVec
                slopeVec = np.array(slope_dict[keyOI])
                whichgenotype = np.where(names_Genotypes == genotypeOI)[0][0]
                gcolor = genotype_colors[whichgenotype,:]
                axes[numrows-3,condIndx].scatter(slopeVec, corrcoeffVec, 
                                                 c = np.tile(gcolor,(len(corrcoeffVec),1)), 
                                                 s = sVec)
                axes[numrows-2,condIndx].hist(corrcoeffVec, density = False, 
                                              color = gcolor.tolist(), alpha = 0.4)
                axes[numrows-1,condIndx].hist(slopeVec, density = False,
                                              color = gcolor.tolist(), alpha = 0.4)
            axes[numrows-3,condIndx].set_xlabel('slope')
            axes[numrows-3,condIndx].set_ylabel('correlation coefficient')
            axes[numrows-3,condIndx].legend(Genotypes_currcond)
            axes[numrows-2,condIndx].set_xlabel('correlation coefficient')
            axes[numrows-2,condIndx].set_ylabel('P')
            axes[numrows-1,condIndx].set_xlabel('slope')
            axes[numrows-1,condIndx].set_ylabel('P')
            # axes[numrows-2,condIndx].legend(Genotypes_currcond)
            
    # plot summary statistics comparing across conditions (for the same genotype)
    if ifcalcorr == True:
        for genotypeIndx in range(numGenotypes):
            if numGenotypes > 1:
                genotypeOI = names_Genotypes[genotypeIndx]
                conds_currg = df[df.genotype == genotypeOI].condition.unique()
            else:
                conds_currg = names_cond
                
            for kk in range(len(conds_currg)):
                condOI = conds_currg[kk]
                if numGenotypes > 1:
                    keyOI = (genotypeOI,condOI)
                else:
                    keyOI = condOI
                corrcoeffVec = corrcoeff_dict[keyOI]
                numsamplesVec = np.array(numsamples_dict[keyOI])
                sVec = minmksize + numsamplesVec
                slopeVec = slope_dict[keyOI]
                whichcond = np.where(names_cond == condOI)[0][0]
                condcolor = cond_colors[whichcond,:]
                axes[genotypeIndx,numcols-3].scatter(slopeVec, corrcoeffVec, 
                                                 c = np.tile(condcolor,(len(corrcoeffVec),1)), 
                                                 s = sVec)
                axes[genotypeIndx,numcols-2].hist(corrcoeffVec, density = False,
                                              color = condcolor.tolist(), alpha = 0.5)
                axes[genotypeIndx,numcols-1].hist(slopeVec, density = False,
                                          color = condcolor.tolist(), alpha = 0.5)

            axes[genotypeIndx,numcols-3].set_xlabel('slope')
            axes[genotypeIndx,numcols-3].set_ylabel('correlation coefficient')
            axes[genotypeIndx,numcols-3].legend(conds_currg)
            axes[genotypeIndx,numcols-2].set_xlabel('correlation coefficient')
            axes[genotypeIndx,numcols-2].set_ylabel('P')
            axes[genotypeIndx,numcols-1].set_xlabel('slope')
            axes[genotypeIndx,numcols-1].set_ylabel('P')
            
# This is a function that plots probability of taking a long trip as a function of x variable of interest (e.g. trip index)
def plotPlongtrip(per_trip_df, xVar, xBinEdges_all, iflogx = False, minmksize = 2, ifshowNsamples = True, rounding = 2, colwidth = 5, 
                      rowwidth = 5, xlabeltype = 'binmid', auto_xBin = False, autoNumxbins = 4, fn2save = None):
    names_cond = per_trip_df.condition.unique()
    numconds = len(names_cond)
    
    xname = xVar
    if iflogx == True:
        xname = 'log10(' + xname + ')'
    yname = 'P(longtrip)'
    
#     f,ax=plt.subplots(1,1,figsize=(colwidth,rowwidth))
    fig = plt.figure(figsize=(colwidth,rowwidth)) #, dpi=300
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        conds_per_trip_df = per_trip_df.loc[(per_trip_df.condition == cond)]
        if auto_xBin == False:
            xBinEdges = xBinEdges_all[condIndx]
        
        rel_df = conds_per_trip_df
        xvals_all = rel_df[xVar].values
        scaledtripdurs_all = rel_df['scaledTripDuration2'].values
        scaledtripdurs_all = scaledtripdurs_all[~np.isnan(xvals_all)]
        xvals_all = xvals_all[~np.isnan(xvals_all)]
        
        if auto_xBin == False:
            numxbins = len(xBinEdges)-1
        else:
            numxbins = np.minimum(len(xvals_all),autoNumxbins)
            quantileOI =  np.array(range(numxbins+1))/numxbins
            xBinEdges = np.quantile(xvals_all,quantileOI)
        xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
        bin_indices = np.digitize(xvals_all, xBinEdges)
        PlongVec = []
        xnames = []
        nsamplesVec = []
        for kk in range(numxbins):
            scaledtripdurs_rel = scaledtripdurs_all[bin_indices==(kk+1)]
            nsamples_rel = len(scaledtripdurs_rel)
            PlongVec.append(np.sum(scaledtripdurs_rel>1.0)/nsamples_rel)
            nsamplesVec.append(nsamples_rel)
            if xlabeltype == 'binmid':
                xnames.append(str(round(xBinMid[kk],rounding))) 
            elif xlabeltype == 'binrange':
                xnames.append(str(round(xBinEdges[kk],rounding)) + '-' + str(round(xBinEdges[kk+1],rounding))) 

        sVec = minmksize + np.array(nsamplesVec)
        if iflogx == False:
            xVec = xBinMid
        else:
            xVec = np.log10(xBinMid)
        plt.scatter(xVec, PlongVec, s = sVec, marker = 'o', alpha = 0.5, label = cond)
        plt.legend(loc='best')
#         plt.legend(loc='upper left')
        axcurr = plt.gca()
#             axcurr.set_xticklabels(xnames)
        axcurr.set_xticks(xVec)
        axcurr.set_ylabel(yname)
        axcurr.set_xlabel(xname)
            
        if ifshowNsamples == True:
            if auto_xBin == False:
                for kk in range(numxbins):
                    axcurr.text(xVec[kk], PlongVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                                              ha='center', va='bottom', color='k')
            else:
                axcurr.text(xVec[0], PlongVec[0] - 0.05, 'n = ' + str(nsamplesVec[0]), 
                                  ha='center', va='bottom', color='k')
                
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

# This is a second function that plots probability of taking a long trip as a function of x variable of interest (e.g. trip index) that makes use of trip type indicator 
def plotPlongtrip_v2(per_trip_df, xVar, xBinEdges_all, iflogx = False, minmksize = 2, ifshowNsamples = True, 
                     xminhard = None, xmaxhard = None, rounding = 2, colwidth = 5, rowwidth = 5, xlabeltype = 'binmid', 
                     auto_xBin = False, autoNumxbins = 4, numxticks = 5, fn2save = None):
    names_cond = per_trip_df.condition.unique()
    numconds = len(names_cond)
    
    xname = xVar
    if iflogx == True:
        xname = 'log10(' + xname + ')'
    yname = 'P(longtrip)'
    
    if auto_xBin == True:
        xvals_allconds = per_trip_df[xVar].values
        xvals_allconds = xvals_allconds[~np.isnan(xvals_allconds)]
        if iflogx == True:
            xminEdge = np.log10(np.min(xvals_allconds[xvals_allconds>0]))
            xmaxEdge = np.log10(np.max(xvals_allconds[xvals_allconds>0]))
        else:
            xminEdge = np.min(xvals_allconds[xvals_allconds>0])
            xmaxEdge = np.max(xvals_allconds[xvals_allconds>0])
        xminEdge = round(xminEdge,rounding)
        xmaxEdge = round(xmaxEdge,rounding)
    else:
        xminEdge = np.min(xBinEdges_all)
        xmaxEdge = np.max(xBinEdges_all)
    if xminhard != None:
        xminEdge = np.maximum(xminEdge,xminhard)
    if xmaxhard != None:
        xmaxEdge = np.minimum(xmaxEdge,xmaxhard)
    xticksVec = np.round(np.linspace(xminEdge,xmaxEdge,numxticks),rounding)
    
#     f,ax=plt.subplots(1,1,figsize=(colwidth,rowwidth))
    fig = plt.figure(figsize=(colwidth,rowwidth)) #, dpi=300
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        conds_per_trip_df = per_trip_df.loc[(per_trip_df.condition == cond)]
        if auto_xBin == False:
            xBinEdges = xBinEdges_all[condIndx]
        
        rel_df = conds_per_trip_df
        xvals_all = rel_df[xVar].values
        triptype_all = rel_df['triptype'].values
        triptype_all = triptype_all[~np.isnan(xvals_all)]
        xvals_all = xvals_all[~np.isnan(xvals_all)]
        
        if auto_xBin == False:
            numxbins = len(xBinEdges)-1
        else:
            numxbins = np.minimum(len(xvals_all),autoNumxbins)
            quantileOI =  np.array(range(numxbins+1))/numxbins
            xBinEdges = np.quantile(xvals_all,quantileOI)
        xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
        bin_indices = np.digitize(xvals_all, xBinEdges)
        PlongVec = []
        xnames = []
        nsamplesVec = []
        for kk in range(numxbins):
            triptype_rel = triptype_all[bin_indices==(kk+1)]
            nsamples_rel = len(triptype_rel)
            PlongVec.append(np.sum(triptype_rel)/nsamples_rel)
            nsamplesVec.append(nsamples_rel)
            if xlabeltype == 'binmid':
                xnames.append(str(round(xBinMid[kk],rounding))) 
            elif xlabeltype == 'binrange':
                xnames.append(str(round(xBinEdges[kk],rounding)) + '-' + str(round(xBinEdges[kk+1],rounding))) 

        sVec = minmksize + np.array(nsamplesVec)
        if iflogx == False:
            xVec = xBinMid
        else:
            xVec = np.log10(xBinMid)
        plt.scatter(xVec, PlongVec, s = sVec, marker = 'o', alpha = 0.5, label = cond)
        plt.legend(loc='best')
#         plt.legend(loc='upper left')
        axcurr = plt.gca()
#             axcurr.set_xticklabels(xnames)
        axcurr.set_xticks(xticksVec)
        axcurr.set_ylabel(yname)
        axcurr.set_xlabel(xname)
            
        if ifshowNsamples == True:
            if auto_xBin == False:
                for kk in range(numxbins):
                    axcurr.text(xVec[kk], PlongVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                                              ha='center', va='bottom', color='k')
            else:
                axcurr.text(xVec[0], PlongVec[0] - 0.05, 'n = ' + str(nsamplesVec[0]), 
                                  ha='center', va='bottom', color='k')
                
    if fn2save != None:
        fig.savefig(fn2save, bbox_inches='tight', dpi=300)

            
# This is a function that plots probability of returning to the food spot (without
# first hitting the boundary wall) given that the fly has experienced a certain 
# x variable (i.e. x > xthres, where x can be displacement from food spot, 
# duration of trip so far, etc.)  
def plotPreturn_varyx(per_trip_df, xVar, xthresVec, ifcalstats = False,
                      rounding = 2, colwidth = 5, rowwidth = 5, 
                      stattest = "Mann-Whitney", correctiontype = None, printstats = False,
                      fn2save = None):
    
    Preturn_df = CreatePreturnDF(per_trip_df, xVar, xthresVec)
    
    names_cond = Preturn_df.condition.unique()
    numconds = len(names_cond)
    
    numxbins = len(xthresVec)
    binnames = [str(i) for i in range(numxbins)]
    xnames = []
    for kk in range(numxbins):
        xname = str(round(xthresVec[kk],rounding))
        xnames.append(xname)
            
    
    f,ax=plt.subplots(1,numconds,figsize=(numconds*colwidth,rowwidth))
    
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        perCond_df = Preturn_df.loc[(Preturn_df.condition == cond)]
            
        gnames = perCond_df.genotype.unique()
        
        boxplot_params = {'data': perCond_df, 'x': 'xGroup', 'y': 'Preturn', 
                                  'order': binnames} #, showfliers: False
        if len(gnames)>1:
            boxplot_params['hue'] = 'genotype'
            
            if ifcalstats == True: # calculate statistical significance 
                pairs = []
                        
                # between genotypes for the same x bin:
                for xbinIndx in range(numxbins):
                    nullmodel = (binnames[xbinIndx],gnames[0])
                    for gIndx in range(len(gnames)-1):
                        model2compare = (binnames[xbinIndx],gnames[gIndx+1])
                        pairs.append([nullmodel,model2compare])

        axcurr = ax[condIndx]    
        sns.boxplot(ax = axcurr, **boxplot_params)
        if (ifcalstats == True) & (len(gnames)>1):
            annotator = Annotator(axcurr, pairs, **boxplot_params)
            annotator.configure(test = stattest,comparisons_correction = correctiontype,verbose = printstats).apply_and_annotate()
            
            
        axcurr.set_xticklabels(xnames)
        axcurr.set_ylabel('Preturn (no wall)')
        if len(gnames)>1:
            axcurr.legend(loc = 'upper right')
        axcurr.set_title(xVar + '(' + cond + ')')
                   
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)
        
# This is a function that plots probability of returning to the food spot (without
# first hitting the boundary wall) as a function of a specified x variable (e.g. meanRL, etc.) given
# specified criteria (e.g. when another variable falls within a certain range). Here, each data point 
# will be an average across all trips that satisfies the desired criteria across all flies with the 
# condition. 
def plotPreturn_varyx_specifycriteria(per_trip_df, xVar, xBinEdges_all, criteria, ifshowNsamples = False, minmksize = 2,
                      rounding = 2, colwidth = 5, rowwidth = 5, xlabeltype = 'binmid', auto_xBin = False, autoNumxbins = 4, fn2save = None):
    
    names_cond = per_trip_df.condition.unique()
    numconds = len(names_cond)
    numcols = len(criteria)
    
    xname = xVar
    yname = 'Preturn'
    
    f,ax=plt.subplots(numconds,numcols,figsize=(numcols*colwidth,numconds*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        conds_per_trip_df = per_trip_df.loc[(per_trip_df.condition == cond)]
        if auto_xBin == False:
            xBinEdges_cond = xBinEdges_all[condIndx]
        
        for keyIndx in range(numcols):
            
            rel_df = conds_per_trip_df
            title = cond + '('
            for key in criteria[keyIndx].keys():
                minvalue = criteria[keyIndx][key]['min']
                maxvalue = criteria[keyIndx][key]['max']
                rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                title = title + (str(round(minvalue,-int(np.floor(np.log10(abs(minvalue)))))) + '<' + key + '<'
                        + str(round(maxvalue,-int(np.floor(np.log10(abs(maxvalue)))))) + ',')
            title = title + ')'
    
            xvals_all = rel_df[xVar].values
            ifloop_all = rel_df['ifloop'].values
            ifloop_all = ifloop_all[~np.isnan(xvals_all)]
            xvals_all = xvals_all[~np.isnan(xvals_all)]
        
            if auto_xBin == False:
                xBinEdges = xBinEdges_cond[keyIndx]
                numxbins = len(xBinEdges)-1
            else:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            bin_indices = np.digitize(xvals_all, xBinEdges)
            PrVec = []
            xnames = []
            nsamplesVec = []
            for kk in range(numxbins):
                ifloop_rel = ifloop_all[bin_indices==(kk+1)]
                if len(ifloop_rel) > 0:
                    PrVec.append(np.sum(ifloop_rel)/len(ifloop_rel))
                else:
                    PrVec.append(np.nan)
                nsamplesVec.append(len(ifloop_rel))
                if xlabeltype == 'binmid':
                    xnames.append(str(round(xBinMid[kk],rounding))) 
                elif xlabeltype == 'binrange':
                    xnames.append(str(round(xBinEdges[kk],rounding)) + '-' + str(round(xBinEdges[kk+1],rounding))) 
            
            
            if (numcols > 1) & (numconds > 1):
                axcurr = ax[condIndx,keyIndx]
            elif (numcols == 1):
                axcurr = ax[condIndx]
            elif (numconds == 1):
                axcurr = ax[keyIndx]
            
            sVec = minmksize + np.array(nsamplesVec)
            axcurr.scatter(xBinMid, PrVec, s = sVec, marker = 'o', alpha = 0.5)
#             axcurr.set_xticklabels(xnames)
            axcurr.set_ylabel(yname)
            axcurr.set_xlabel(xname)
            axcurr.set_title(title)
            
            if ifshowNsamples == True:
                for kk in range(numxbins):
                    axcurr.text(xBinMid[kk], PrVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                                              ha='center', va='bottom', color='k')
                # if auto_xBin == False:
                #     for kk in range(numxbins):
                #         axcurr.text(xBinMid[kk], PrVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                #                                   ha='center', va='bottom', color='k')
                # else:
                #     axcurr.text(xBinMid[0], PrVec[0] - 0.05, 'n = ' + str(nsamplesVec[0]), 
                #                       ha='center', va='bottom', color='k')
        
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)
    

# This is a function for plotting how the probability of an event varies as a function of a specified x variable given
# specified criteria (e.g. when another variable falls within a certain range). 
# The event here can be returning to food spot (i.e. if trip is a loop), whether the fly turned in preferred direction, etc.
# This is a generalization of the previous 'plotPreturn_varyx_specifycriteria' to allow more general applications (when we are  
# interested in averaging a binary variable i.e. the fraction of samples where the variable is 1)
def plotPevent_varyx_specifycriteria(df, yVar, xVar, xBinEdges_all, criteria, ifshowNsamples = False, minmksize = 2, sizefactor = 1,
              rounding = 2, colwidth = 5, rowwidth = 5, ifoverlap = False, xlabeltype = 'binmid', auto_xBin = False, autoNumxbins = 4, 
              fitparamsFuncType = None, fn2save = None):
    
    names_cond = df.condition.unique()
    numconds = len(names_cond)
    numcols = len(criteria)
    if ifoverlap == False:
        numrows = numconds
    else:
        numrows = 1
        legend = []
        for condIndx in range(numconds):
            cond = names_cond[condIndx]
            legend.append('condition: ' + cond)
    
    xname = xVar
    yname = 'P(' + yVar + ')'
                    
    f,ax=plt.subplots(numrows,numcols,figsize=(numcols*colwidth,numrows*rowwidth))
    for condIndx in range(numconds):
        cond = names_cond[condIndx]
        conds_df = df.loc[(df.condition == cond)]
            
        if auto_xBin == False:
            xBinEdges_cond = xBinEdges_all[condIndx]
        
        for keyIndx in range(numcols):
            
            rel_df = conds_df
            title = cond + '('
            for key in criteria[keyIndx].keys():
                minvalue = criteria[keyIndx][key]['min']
                maxvalue = criteria[keyIndx][key]['max']
                rel_df = rel_df.loc[(rel_df[key] >= minvalue) & (rel_df[key] <= maxvalue)]
                title = title + (str(round(minvalue,-int(np.floor(np.log10(abs(minvalue)))))) + '<' + key + '<'
                        + str(round(maxvalue,-int(np.floor(np.log10(abs(maxvalue)))))) + ',')
            title = title + ')'
    
            xvals_all = rel_df[xVar].values
            ifevent_all = rel_df[yVar].values
            ifevent_all = ifevent_all[~np.isnan(xvals_all)]
            xvals_all = xvals_all[~np.isnan(xvals_all)]
        
            if auto_xBin == False:
                xBinEdges = xBinEdges_cond[keyIndx]
                numxbins = len(xBinEdges)-1
            else:
                numxbins = np.minimum(len(xvals_all),autoNumxbins)
                quantileOI =  np.array(range(numxbins+1))/numxbins
                xBinEdges = np.quantile(xvals_all,quantileOI)
            xBinMid = (xBinEdges[0:-1]+xBinEdges[1:])/2
            bin_indices = np.digitize(xvals_all, xBinEdges)
            PrVec = []
            xnames = []
            nsamplesVec = []
            for kk in range(numxbins):
                ifevent_rel = ifevent_all[bin_indices==(kk+1)]
                if len(ifevent_rel) > 0:
                    PrVec.append(np.sum(ifevent_rel==1)/(np.sum(ifevent_rel==1)+np.sum(ifevent_rel==0)))
                    # PrVec.append(np.sum(ifevent_rel)/len(ifevent_rel))
                else:
                    PrVec.append(np.nan)
                nsamplesVec.append(len(ifevent_rel))
                if xlabeltype == 'binmid':
                    xnames.append(str(round(xBinMid[kk],rounding))) 
                elif xlabeltype == 'binrange':
                    xnames.append(str(round(xBinEdges[kk],rounding)) + '-' + str(round(xBinEdges[kk+1],rounding))) 
            PrVec = np.array(PrVec)
            nsamplesVec = np.array(nsamplesVec)
            
            if (numcols > 1) & (numrows > 1):
                axcurr = ax[condIndx,keyIndx]
            elif (numcols == 1) & (numrows > 1):
                axcurr = ax[condIndx]
            elif (numrows == 1):
                if numcols > 1:
                    axcurr = ax[keyIndx]
                else:
                    axcurr = ax

            xBinMid = xBinMid[~np.isnan(PrVec)]
            nsamplesVec = nsamplesVec[~np.isnan(PrVec)]
            PrVec = PrVec[~np.isnan(PrVec)]
            sVec = minmksize + sizefactor*nsamplesVec
            # sVec = minmksize + sizefactor*np.array(nsamplesVec)
            axcurr.scatter(xBinMid, PrVec, s = sVec, marker = 'o', alpha = 0.5)
            if fitparamsFuncType != None:
                if (fitparamsFuncType == 'PLfunc') or (fitparamsFuncType == 'ExpFunc'):
                    if PrVec[1] < PrVec[0]:
                        fitFuncType = 'decreasing' + fitparamsFuncType
                    else:
                        fitFuncType = 'increasing' + fitparamsFuncType
                else:
                    fitFuncType = fitparamsFuncType
                
                # extract initial guess and bounds for function parameters:
                param_guess, bounds = GetParamGuessAndBounds(xBinMid, PrVec, fitFuncType)
                parameters, covariance = curve_fit(eval(fitFuncType), xBinMid, PrVec, param_guess, bounds = bounds, maxfev=5000)
                # (parameters, covariance, infodict, mesg, ier) = curve_fit(eval(fitFuncType), xdata, ydata, param_guess, full_output=True)
                # print(infodict['fvec'])
                xscan = np.linspace(np.min(xBinMid),np.max(xBinMid),1000) 
                axcurr.plot(xscan, eval(fitFuncType)(xscan,*parameters), '-', label='fit')
                # Get equation name
                EqnName = GetEqnString(fitFuncType,parameters)
                axcurr.text((np.min(xscan)+np.max(xscan))/2,(np.min(PrVec)+np.max(PrVec))/2,EqnName,fontsize ='medium')

#             axcurr.set_xticklabels(xnames)
            axcurr.set_ylabel(yname)
            axcurr.set_xlabel(xname)
            axcurr.set_title(title)
            
            if ifshowNsamples == True:
                for kk in range(numxbins):
                    axcurr.text(xBinMid[kk], PrVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                                              ha='center', va='bottom', color='k')
                # if auto_xBin == False:
                #     for kk in range(numxbins):
                #         axcurr.text(xBinMid[kk], PrVec[kk] - 0.05, 'n = ' + str(nsamplesVec[kk]), 
                #                                   ha='center', va='bottom', color='k')
                # else:
                #     axcurr.text(xBinMid[0], PrVec[0] - 0.05, 'n = ' + str(nsamplesVec[0]), 
                #                       ha='center', va='bottom', color='k')

            # if ifoverlap == True:
            #     axcurr.legend(legend)
            
        
    if fn2save != None:
        f.savefig(fn2save, bbox_inches='tight', dpi=300)


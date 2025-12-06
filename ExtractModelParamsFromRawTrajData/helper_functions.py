# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:58:22 2022

@author: labadmin
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys

from sklearn.mixture import GaussianMixture
from scipy import stats



## used for reading YAML files
def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out

## combine multiple metadata files
def Getmetadata(filenames):
    metadata = read_yaml(filenames[0])
    for kk in range(len(filenames)-1):
        metadata1 = read_yaml(filenames[kk+1])
        metadata.update(metadata1)
    return metadata    

def rle(arr, dt=None):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(arr, dtype=np.int32)                  # force numpy
    n = len(ia)
    if n == 0:
        if dt is None:
            return ([], [], [])
        else:
            return ([], [], [], [])
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe). select time points where there's a change in state
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions

        if dt is None:
            return(z, p, ia[i]) # simply return array runlengths
        else:
            try:
                dt = np.array(dt)   # force numpy
                l = np.zeros(z.shape) ## real time durations
                for j,_ in enumerate(p[:-1]):
                    l[j] = np.sum(dt[p[j]:p[j+1]])
                l[-1] = np.sum(dt[p[-1]:]) ## length of last segment
                return(z, p, ia[i], l) # return array runlengths & real time durations
            except TypeError:
                print('Your array is invalid')

# function for computing turn angle between two angles
def computeTurnAngBetween2angles(startangle,endangle):
    turnAngle = endangle - startangle
    if hasattr(turnAngle, "__len__") == False:
        if turnAngle > np.pi:
            turnAngle =  turnAngle - 2*np.pi
        elif turnAngle <= -np.pi:
            turnAngle = turnAngle + 2*np.pi
    else:
        turnAngle[turnAngle>np.pi] = turnAngle[turnAngle>np.pi] - 2*np.pi
        turnAngle[turnAngle<=-np.pi] = turnAngle[turnAngle<=-np.pi] + 2*np.pi
    return turnAngle


# function for computing turn angle from vec1 to vec2, each vector has coordinations [x,y]:
def computeTurnAngBetween2vectors(vec1,vec2):
    turnSize = np.arctan2(vec2[1],vec2[0]) - np.arctan2(vec1[1],vec1[0])
    if turnSize > np.pi:
        turnSize =  -(2*np.pi - turnSize)
    elif turnSize <= -np.pi:
        turnSize = turnSize + 2*np.pi
    return turnSize

                
# function for fitting data to a Gaussian mixture model
def fitGMM(data,Nmin,Nmax, parameterPenaltyType = 'both', xlabel = 'x', condOI = 'WT_40hr', ifplot = False, numbins = 20,
                      colwidth = 5, rowwidth = 5):

    # fit models with 1-5 components
    N = np.arange(Nmin, Nmax+1)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(data)

    # compute the AIC and the BIC
    if parameterPenaltyType == 'both':
        AIC = [m.aic(data) for m in models]
        BIC = [m.bic(data) for m in models]
    
        # extract best model
        M_best = models[np.min([np.argmin(AIC),np.argmin(BIC)])]
    elif parameterPenaltyType == 'AIC':
        AIC = [m.aic(data) for m in models]
        M_best = models[np.argmin(AIC)]
    elif parameterPenaltyType == 'BIC':
        BIC = [m.bic(data) for m in models]
        M_best = models[np.argmin(BIC)]
    
    if ifplot == True:
        f,ax=plt.subplots(1,2,figsize=(2*colwidth,rowwidth))

        xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 1000)
        logprob = M_best.score_samples(xscan.reshape(-1, 1))
        responsibilities = M_best.predict_proba(xscan.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        ax[0].hist(data, 20, density=True, histtype='stepfilled', alpha=0.6)
        ax[0].plot(xscan, pdf, '-k')
        ax[0].plot(xscan, pdf_individual, '--k')
        # ax[0].text(0.5, np.max(pdf)+0.1, "Best-fit Mixture",ha='left', va='top')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel('P')
        ax[0].set_title(condOI + '(mu = ' + str(np.round(M_best.means_[0][0],2)) + 
                        ', sigma = ' + str(np.round(M_best.covariances_[0][0][0],2)) + ')')

        if (parameterPenaltyType == 'both') or (parameterPenaltyType == 'AIC'):
            ax[1].plot(N, AIC, '-k', label='AIC')
        if (parameterPenaltyType == 'both') or (parameterPenaltyType == 'BIC'):
            ax[1].plot(N, BIC, '--k', label='BIC')
        ax[1].set_xlabel('n. components')
        ax[1].set_ylabel('information criterion')
        ax[1].legend(loc=2)

#     return (M_best.weights_, M_best.means_, M_best.covariances_)
    return M_best

# Function to extract intersects between 2 individual Gaussians in GMM
def extractGaussianIntersects(data):
    model = GaussianMixture(2).fit(data)
    xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 2000)
    logprob = model.score_samples(xscan.reshape(-1, 1))
    responsibilities = model.predict_proba(xscan.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    
    pdf_diff = pdf_individual[:,0] - pdf_individual[:,1]
    signdiff = np.sign(pdf_diff)
    transitionIndx = np.where(signdiff[0:-1] != signdiff[1:])[0]

    xthres = (xscan[transitionIndx] + xscan[transitionIndx+1])/2
    
    return xthres
    
# Function to extract intersects between multiple (n) individual Gaussians in GMM
def extractGaussianIntersects_v2(data,n):
    model = GaussianMixture(n).fit(data)
    xscan = np.linspace(np.min(data)-0.2, np.max(data)+0.2, 2000)
    logprob = model.score_samples(xscan.reshape(-1, 1))
    responsibilities = model.predict_proba(xscan.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    maxInds = np.argmax(pdf_individual,axis=0)
    pdf_individual = pdf_individual[:,maxInds.argsort()]

    xthres_all = []
    for kk in range(n-1):
        pdf_diff = pdf_individual[:,kk] - pdf_individual[:,kk+1]
        signdiff = np.sign(pdf_diff)
        transitionIndx = np.where(signdiff[0:-1] != signdiff[1:])[0]
        if len(transitionIndx) > 1:
            # print(transitionIndx)
            Pvals = pdf_individual[transitionIndx,kk]
            whichIndx = np.argmax(Pvals)
            # print(whichIndx)
            transitionIndx = transitionIndx[whichIndx]
        xthres = (xscan[transitionIndx] + xscan[transitionIndx+1])/2
        # print(xthres)
        xthres_all.append(xthres)
    
    return xthres_all


# define cdf of GMM
def mix_norm_cdf(x, weights, means, sigmas):
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * stats.norm.cdf(x, loc=means[i], scale=sigmas[i])
    return mcdf

# Extract number of conditions and what there are from dataframe
def GetConds(df):
    names_cond = df.condition.unique()
    totnumconds = len(names_cond)   
    return (names_cond,totnumconds)

# Extract total number of flies and #flies in each condition from dataframe
def GetNumFlies(df):
    totnumflies = len(df.fly.unique())
    # number of flies in each condition
    numfliesVec = []
    for condIndx in range(len(df.condition.unique())):
        cond = df.condition.unique()[condIndx]
        numfliesVec.append(len(df[df.condition==cond].fly.unique()))
    
    return (numfliesVec, totnumflies)


def roundToSigFig(x, sig):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def GetAutoCorrelation(data):
   
    # Mean
    mean = np.mean(data)
    
    # Variance
    var = np.var(data)
    
    # Normalized data
    ndata = data - mean
    
    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
    acorr = acorr / var / len(ndata)


# This is a function for automatically getting the names of the parameters of a specified distribution
def list_parameters(distribution):
    """List parameters for scipy.stats.distribution.
    # Arguments
        distribution: a string or scipy.stats distribution object.
    # Returns
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        distribution = getattr(stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]
    else:
        parameters = []
    if distribution.name in stats._discrete_distns._distn_names:
        parameters += ['loc']
    elif distribution.name in stats._continuous_distns._distn_names:
        parameters += ['loc', 'scale']
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters

# This is a function for filtering out relevant parts of the whole dataframe
def GetRelevantDF(originalDF, criteria):
    relDF = originalDF
    for key in criteria.keys():
        relgrps = criteria[key]
        relDF = relDF[np.isin(relDF[key].values,relgrps)]

    return relDF

# This is a function for autobinning a variable such that there are equal number of samples in each bin
# given a specified number of bins. This function outputs the bin edges and middle of bins
def autobin(dataVec,numbins):
    numbins = np.minimum(len(dataVec),numbins)
    quantileOI =  np.array(range(numbins+1))/numbins
    binEdges = np.quantile(dataVec,quantileOI)
    binMid = (binEdges[0:-1]+binEdges[1:])/2
        
    return (binEdges, binMid)


# functional forms for fitting
def powerlawFunc(x,y0,alpha,beta):
    y = y0*(1+alpha*x**beta)
    return y

def increasingPLfunc(x,y0,ymax,alpha,beta):
    y = ymax - (ymax-y0)/((1+alpha*x)**beta)
    return y

def decreasingPLfunc(x,y0,ymin,alpha,beta):
    y = ymin + (y0-ymin)/((1+alpha*x)**beta)
    return y
    
def increasingExpFunc(x,y0,ymax,alpha):
    y = ymax - (ymax-y0)*np.exp(-alpha*x)
    return y

def decreasingExpFunc(x,y0,ymin,alpha):
    y = ymin + (y0-ymin)*np.exp(-alpha*x)
    return y

def sumTwoExpFunc(x,A,alpha,B,beta):
    y = A*np.exp(-alpha*x) + B*(np.exp(beta*x)-1)
    return y

# An alternative sum of two exponentials with an initially increasing exponential and then a decreasing one
def sumTwoExpFunc_v2(x,k,A,alpha,B,beta): 
    y = k - A*np.exp(-alpha*x) + B*np.exp(beta*x)
    return y
    


# function that outputs estimated values/guesses and bounds for parameters of function to be fitted to data
def GetParamGuessAndBounds(xdata, ydata, fitFuncType, y0known = None):

    # extract estimates for initial values of y, dy/dt, d2y/dt2
    yinit_est = ydata[0]
    ymax_est = np.max(ydata)
    ymin_est = np.min(ydata)
    dydtinit_est = (ydata[1]-ydata[0])/(xdata[1]-xdata[0])
    d2ydt2init_est = ((ydata[2]-ydata[1])/(xdata[2]-xdata[1]) - (ydata[1]-ydata[0])/(xdata[1]-xdata[0]))/((xdata[2]-xdata[0])/2)

    if y0known == None:
        y0_guess = yinit_est
    ymin_guess = ymin_est
    ymax_guess = ymax_est
    if fitFuncType == 'decreasingExpFunc':
        alpha_guess = -dydtinit_est/(y0_guess-ymin_guess)
        param_guess = [y0_guess, ymin_guess, alpha_guess]
        lbvec = np.array([-np.inf, -np.inf, 0])
        ubvec = np.ones((3,))*np.inf
        bounds = (lbvec,ubvec)
        # bounds = [(0, np.inf), (0, np.inf), (0, np.inf)]
    elif fitFuncType == 'increasingExpFunc':
        alpha_guess = dydtinit_est/(ymax_guess-y0_guess)
        param_guess = [y0_guess, ymax_guess, alpha_guess]
        lbvec = np.array([-np.inf, -np.inf, 0])
        ubvec = np.ones((3,))*np.inf
        bounds = (lbvec,ubvec)
        # bounds = [(0, np.inf), (0, np.inf), (0, np.inf)]
    elif fitFuncType == 'decreasingPLfunc':
        ab_guess = -dydtinit_est/(y0_guess-ymin_guess)
        a_guess = np.abs(d2ydt2init_est/(ab_guess*(y0_guess-ymin_guess)) - ab_guess)
        b_guess = np.abs(ab_guess/a_guess)
        param_guess = [y0_guess, ymin_guess, a_guess, b_guess]
        lbvec = np.array([-np.inf, -np.inf, 0, 0])
        ubvec = np.ones((4,))*np.inf
        bounds = (lbvec,ubvec)
        # bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    elif fitFuncType == 'increasingPLfunc':
        ab_guess = dydtinit_est/(ymax_guess-y0_guess)
        a_guess = np.abs(-d2ydt2init_est/(ab_guess*(ymax_guess-y0_guess)) - ab_guess)
        b_guess = np.abs(ab_guess/a_guess)
        param_guess = [y0_guess, ymax_guess, a_guess, b_guess]
        lbvec = np.array([-np.inf, -np.inf, 0, 0])
        ubvec = np.ones((4,))*np.inf
        bounds = (lbvec,ubvec)
        # bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    elif fitFuncType == 'sumTwoExpFunc':
        xlast = xdata[-1]
        ylast = ydata[-1]
        dydtlast_est = (ydata[-1]-ydata[-2])/(xdata[-1]-xdata[-2])
    
        A_guess = y0_guess
        alpha_guess = -dydtinit_est/A_guess
        beta_guess = dydtlast_est/ylast - 1/xlast
        B_guess = (ylast/xlast)/beta_guess
        param_guess = [A_guess, alpha_guess, B_guess, beta_guess]
        lbvec = -np.ones((4,))*np.inf
        ubvec = np.ones((4,))*np.inf
        bounds = (lbvec,ubvec)

    elif fitFuncType == 'sumTwoExpFunc_v2':
        k_guess = ydata[-1]
        A_guess = k_guess/2
        B_guess = A_guess + y0_guess - k_guess

        dydtlast_est = (ydata[-1]-ydata[-2])/(xdata[-1]-xdata[-2])
        d2ydt2init_est = ((ydata[-1]-ydata[-2])/(xdata[-1]-xdata[-2]) - (ydata[-2]-ydata[-3])/(xdata[-2]-xdata[-3]))/((xdata[-1]-xdata[-3])/2)
        beta_guess = -d2ydt2init_est/dydtlast_est
        alpha_guess = (dydtinit_est + B_guess*beta_guess)/A_guess
        
        param_guess = [k_guess, A_guess, alpha_guess, B_guess, beta_guess]
        lbvec = -np.ones((4,))*np.inf
        ubvec = np.ones((4,))*np.inf
        bounds = (lbvec,ubvec)
        
    return (param_guess,bounds)

# This is a function for defining name of equation (e.g. for printing on figure)
# given the equation type and the values of the parameters
def GetEqnString(fitFuncType,paramsVec,sf=3):
    if fitFuncType == 'decreasingExpFunc':
        y0 = paramsVec[0]
        ymin = paramsVec[1]
        alpha = paramsVec[2]
        EqnName = (r"$y =  " + str(roundToSigFig(ymin,sf)) + "+ " +  
                   str(roundToSigFig(y0 - ymin,sf)) + " e^{- " + str(roundToSigFig(alpha,sf)) + "x}$")
    elif fitFuncType == 'increasingExpFunc':
        y0 = paramsVec[0]
        ymax = paramsVec[1]
        alpha = paramsVec[2]
        EqnName = (r"$y =  " + str(roundToSigFig(ymax,sf)) + "- " +  
                   str(roundToSigFig(ymax - y0,sf)) + " e^{- " + str(roundToSigFig(alpha,sf)) + "x}$")
    elif fitFuncType == 'decreasingPLfunc':
        y0 = paramsVec[0]
        ymin = paramsVec[1]
        alpha = paramsVec[2]
        beta = paramsVec[3]
        # EqnName = r"$y =  " + str(ymin) + "+ \frac{" +  str(y0 - ymin) + "}{(1 + " + str(alpha) + "x)^" + str(beta) + "}$"
        EqnName = (r"$y =  " + str(roundToSigFig(ymin,sf)) + "+ " +  str(roundToSigFig(y0 - ymin,sf)) + 
           "/(1 + " + str(roundToSigFig(alpha,sf)) + "x)^{" + str(roundToSigFig(beta,sf)) + "}$")

    elif fitFuncType == 'increasingPLfunc':
        y0 = paramsVec[0]
        ymax = paramsVec[1]
        alpha = paramsVec[2]
        beta = paramsVec[3]
        # EqnName = r"$y =  " + str(ymax) + "- \frac{" +  str(ymax - y0) + "}{(1 + " + str(alpha) + "x)^" + str(beta) + "}$"
        EqnName = (r"$y =  " + str(roundToSigFig(ymax,sf)) + "- " +  str(roundToSigFig(ymax - y0,sf)) + 
           "/(1 + " + str(roundToSigFig(alpha,sf)) + "x)^{" + str(roundToSigFig(beta,sf)) + "}$")

    elif fitFuncType == 'sumTwoExpFunc':
        A = paramsVec[0]
        alpha = paramsVec[1]
        B = paramsVec[2]
        beta = paramsVec[3]
        EqnName = (r"$y =  " + str(roundToSigFig(A,sf)) + "e^{- " +  str(roundToSigFig(alpha,sf)) + "x} +" + 
           str(roundToSigFig(B,sf)) + "(e^{" +  str(roundToSigFig(beta,sf)) + "x} - 1)$")

    elif fitFuncType == 'sumTwoExpFunc_v2':
        k = paramsVec[0]
        A = paramsVec[1]
        alpha = paramsVec[2]
        B = paramsVec[3]
        beta = paramsVec[4]
        EqnName = (r"$y =  " + str(roundToSigFig(k,sf)) + '-' + str(roundToSigFig(A,sf)) + "e^{- " +  str(roundToSigFig(alpha,sf)) + "x} +" + 
           str(roundToSigFig(B,sf)) + "e^{" +  str(roundToSigFig(beta,sf)) + "x} $")
    

    return EqnName
        
        
    

                
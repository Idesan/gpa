# -*- coding: utf-8 -*-
"""
Utility library for GPA (Generative Perturbation Analysis)

Last updated on Feb 05, 2023

@author: Tsuyoshi Ide (ide@ide-research.net / tide@us.ibm.com )
"""

import numpy as np
import pandas as pd
import scipy as sp

from sklearn import linear_model
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_array

import matplotlib.pyplot as plt
import seaborn as sb; sb.set()

def IG_vec(xtest,xbase,model,eta,N_alpha =100,N_grad=10,seed=0,h_minimum=1.e-8):
    '''
    Vectorized computation of Integrated Gradient (IG) based on IG_vec_i()

    Parameters
    ----------
    xtest : array-like
        Test point coordinate. Only a single point (ndim=1) is allowed.
    xbase : array-like
        baseline point coordinate. Only a single point (ndim=1) is allowed.
    model : TYPE
        An instance method predict () must be available.
    eta : float
        Standard deviation of Gaussian perturbations for gradient estimation.
    N_alpha : int, optional
        The number of bins for numerical integration of alpha in the definition
        of IG. The trapezoidal method is used. Note that the number of the grid
        points becomes N_alpha+1. The default is 100.
    N_grad : int, optional
        The number of random perturbations in gradient estimation.
    seed : int, optional
        Random seed for the perturbations in gradient estimation. The default is 0.
    h_minimum : float, optional
        The minimum scale of perturbation. This is to prevent division
        by 0. In the code, eta*h_minimum is used for the lower threshold.
        The default is 1.e-8.

    Returns
    -------
    IG : 1D ndarray
        Integrated gradient value for each dimension.

    '''
    # Ensure to have an ndarray of float.
    xtest = np.array(xtest).astype(float).ravel()
    xbase = np.array(xbase).astype(float).ravel()


    M = len(xtest)
    IG = np.empty(M)
    for ii in range(M):
        IG[ii] = IG_vec_i(idx=ii, xtest=xtest,xbase=xbase, model= model,
                             eta=eta, N_alpha =N_alpha, N_grad=N_grad,
                             seed=seed, h_minimum=h_minimum)
    return IG


def IG_vec_i(idx, xtest,xbase,model,eta,N_alpha =100,N_grad=10,seed=0,
               h_minimum=1.e-8):
    '''
    Compute the Integrated Gradient for one specific dimension.
    This method is loop-free (but memory-hungry).

    Parameters
    ----------
    idx : int
        The index for which IG is computed.
    xtest : array-like
        Test point coordinate. Only a single point (ndim=1) is allowed.
    xbase : array-like
        baseline point coordinate. Only a single point (ndim=1) is allowed.
    model : TYPE
        An instance method predict () must be available.
    eta : float
        Standard deviation of Gaussian perturbations for gradient estimation.
    N_alpha : int, optional
        The number of bins for numerical integration of alpha in the definition
        of IG. The trapezoidal method is used. Note that the number of the grid
        points becomes N_alpha+1. The default is 100.
    N_grad : int, optional
        The number of random perturbations in gradient estimation.
    seed : int, optional
        Random seed for the perturbations in gradient estimation. The default is 0.
    h_minimum : float, optional
        The minimum scale of perturbation. This is to prevent division
        by 0. In the code, eta*h_minimum is used for the lower threshold.
        The default is 1.e-8.

    Returns
    -------
    IG_i : float
        IG value for the dimension.

    '''


    import numpy as np

    x_test = np.array(xtest).astype(float).ravel()
    x_base = np.array(xbase).astype(float).ravel()

    da = 1./N_alpha
    alphas = da* np.arange(0,N_alpha+1)

    # Create the Z-matrix
    dx = (x_test-x_base).reshape(1,-1)
    Z = alphas.reshape(-1,1).dot(dx) +  x_base.reshape(1,-1)


    # Create the hh vector
    h0 = np.random.default_rng(seed).normal(0,eta,N_grad)

    h = h0[abs(h0) >= h_minimum*eta] # Drop too small perturbations
    NN_grad = len(h)

    hh = np.zeros([N_alpha+1,N_grad]) + h
    hh = hh.flatten(order='F')

    # Create ZB and evaluate f(ZB)
    ZB = np.tile(Z,[NN_grad,1])
    ff0 = model.predict(ZB)

    # Add perturbations to the i-column of ZB, and evaluate f().
    ZB[:,idx] = ZB[:,idx] + hh
    ff = model.predict(ZB)

    # IG: numerical integration (0-th term of trapezoidal rule)
    # This term correponds to the rectangular rule
    # (1/N_alpha) is not for avaraging but the bin size.
    term1 = da*( (ff - ff0)/hh).sum()/NN_grad

    # Correction terms of trapezoidal rule, corresponding to
    # the left and right terminal bins
    Z0 = np.tile(x_base.reshape(1,-1),[NN_grad,1])
    f0 = model.predict(Z0)
    Z0[:,idx] = Z0[:,idx] + h
    f1 = model.predict(Z0)
    term0 = da*( (f1 - f0)/h ).sum()/NN_grad

    ZN = np.tile(x_test.reshape(1,-1),[NN_grad,1])
    f0 = model.predict(ZN)
    ZN[:,idx] = ZN[:,idx] + h
    f1 = model.predict(ZN)
    termN = da*( (f1 - f0)/h ).sum()/NN_grad

    # IG value
    IG_i = (x_test[idx] - x_base[idx])*(term1 - 0.5*term0 - 0.5*termN )

    return IG_i


def z_score(xtest,Xtrain,eps=1.e-10):
    '''
    Z-score.

    Parameters
    ----------
    xtest : 1D ndarray
        DESCRIPTION.
    Xtrain : TYPE
        DESCRIPTION.
    eps : float, optional
        DESCRIPTION. The default is 1.e-10.

    Returns
    -------
    score : 1D ndarray
        The Z score.

    '''
    xmean = Xtrain.mean(axis=0)
    sigma = Xtrain.std(axis=0)
    score = (xtest-xmean)/(sigma + eps)
    return score


def EIG_vec_i(idx, x_test, model, Xtrain, N_alpha =100, N_grad=10,
             eta=0.1, h_minimum = 1e-8, seed=0):
    '''
    Computes the EIG for a specific variable.
    Avoids using loops as many as possible. Should be fast in Python.

    Parameters
    ----------
    idx : TYPE
        The variable index at which EIG is computed.
    x_test : ndarray
        Test point coordinate. Only a single point (ndim=1) is allowed.
    model : TYPE
        An instance method predict () must be available.
    Xtrain : 2D ndarray
        Training data on which the expectation of EIG is computed.
    N_alpha : int, optional
        The number of bins for numerical integration of alpha in the definition
        of IG. The trapezoidal method is used. Note that the number of the grid
        points becomes N_alpha+1. The default is 100.
    N_grad : int, optional
        The number of random perturbations in gradient estimation. The default
        is 10.
    eta : float
        Standard deviation of Gaussian perturbations for gradient estimation.
    h_minimum : float, optional
        The minimum scale of perturbation. This is to prevent division
        by 0. In the code, eta*h_minimum is used for the lower threshold.
        The default is 1.e-8.
    seed : int, optional
        Random seed for the perturbations in gradient estimation. The default is 0.

    Returns
    -------
    EIG_i : float
        Expected Integrated Gradient for the i-th variable.

    '''

    import numpy as np

    # Input must be a Numpy array. No list, no int as the input.
    x_test = np.array(x_test).astype(float).ravel()

    # Grid points for the trapezoidal integral of the alpha parameter of IG.
    # The grit points has numbers from 0 through N_alpha 
    # (i.e., the total grid points is N_alpha +1). 
    dal = 1./N_alpha
    alphas = dal*np.arange(0,N_alpha+1)

    # Baseline position = training data. Hence, IG is NOT doubly black-box.
    X_base = Xtrain
    N_base, M = X_base.shape

    # Pre-populate perturbations for gradient estimation
    h0 = np.random.default_rng(seed).normal(0,eta,N_grad)
    h = h0[abs(h0) >= h_minimum*eta] # Drop too small perturbations. 
    NN_grad = len(h) ## Use this instead of N_grad. NN_grad can be smaller than N_grad.


    #------ The rightmost term in the trapezoid rule except for the (-1/2) prefactor.
    # This term dees not depend on x_base.
    # (x^t-x_base) term 
    termN_A = - (X_base[:,idx] - x_test[idx]).mean(axis=0) 

    # Gradient term
    Z_right = np.tile(x_test.reshape(1,-1),[NN_grad,1])
    f0_right = model.predict(Z_right)
    Z_right[:,idx] = Z_right[:,idx] + h
    f_right = model.predict(Z_right)

    termN = termN_A * dal*( (f_right - f0_right)/h ).sum()/NN_grad


    #----- The leftmost term except for the (-1/2) prefactor. 
    # Does depend on x_base.
    # (x^t-x_base) term
    dx = (x_test[idx] - X_base[:,idx]).reshape(-1,1) # 縦ベクトル
    term0_A = np.tile(dx,[NN_grad,1]).flatten(order='F')/N_base

    # Gradient term
    Z_left = np.tile(X_base, [NN_grad,1])
    hh_left = np.tile(h.reshape(1,-1),[N_base,1]).flatten(order='F')
    f0_left = model.predict(Z_left)
    Z_left[:,idx] = Z_left[:,idx] + hh_left
    f_left = model.predict(Z_left)
    term0_B = dal*( (f_left - f0_left)/hh_left)/NN_grad

    term0 = (term0_A * term0_B).sum()

    #--- For non-terminal points ---

    # (x^t-x_base) term
    dx = (x_test[idx] - X_base[:,idx]).reshape(1,-1) # row vector
    dX = np.tile(dx, [NN_grad*(N_alpha+1),1]) # tall matrix stacking the row vectors
    term_A = dX.flatten(order='F')/N_base # Stacking the tall matrix vertically

    ZB = np.empty([(N_alpha+1)*NN_grad*N_base, M])
    for n in range(N_base):
        x_base = X_base[n,:]
        Zn = alphas.reshape(-1,1).dot(x_test.reshape(1,-1)) \
            + (1 - alphas).reshape(-1,1).dot(x_base.reshape(1,-1))
        ZBn = np.tile(Zn,[NN_grad,1])

        i1 = n*(NN_grad*(N_alpha +1))
        i2 = i1 + NN_grad*(N_alpha+1)
        ZB[i1:i2,:] = ZBn

    # Output before perturbation ---
    fB0 = model.predict(ZB)
    
    # Output after perturbation ----
    # Computing perturbation vector
    hh = np.tile(h.reshape(1,-1),[N_alpha+1,1]).flatten(order='F')
    hhB = np.tile(hh.reshape(-1,1),[N_base,1]).ravel()
    # Computing f(x) with perturbation
    ZB[:,idx] = ZB[:,idx] + hhB
    fB = model.predict(ZB)
    # term_B
    term_B = (dal/NN_grad) * ((fB-fB0)/hhB)

    term = (term_A*term_B).sum()


    #-------- EIG 
    EIG_i = term -0.5*(term0 + termN)

    return EIG_i


def EIG_vec(x_test, model, Xtrain, N_alpha =100, N_grad=10,
             eta=0.1, h_minimum = 1e-8, seed=0):
    '''
    Repeat EIG_vec_i over all i's.

    Parameters
    ----------
    x_test : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    Xtrain : TYPE
        DESCRIPTION.
    N_alpha : TYPE, optional
        DESCRIPTION. The default is 100.
    N_grad : TYPE, optional
        DESCRIPTION. The default is 10.
    eta : TYPE, optional
        DESCRIPTION. The default is 0.1.
    h_minimum : TYPE, optional
        DESCRIPTION. The default is 1e-8.
    seed : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    EIG : 1D array
        EIG score for each variable.

    '''
    x_test = np.array(x_test).astype(float).ravel()

    X_base = Xtrain
    N_base, M = X_base.shape

    EIG = np.empty(M)
    for ii in range(M):
        EIG[ii] = EIG_vec_i(idx=ii, x_test=x_test, model=model,
                        Xtrain=Xtrain, N_alpha =N_alpha, N_grad=N_grad,
                     eta=eta, h_minimum = h_minimum, seed=seed)
    return EIG


def SV(xtest,Xtrain,model,N_samples_used=None,seed=1,reporting_percentage=0):
    '''
    Implementation of Algorithm 1 of [Strumbeli and Kononenko 2014], which computes 
    the Shapley values using a Monte Carlo approximation. 

    Parameters
    ----------
    xtest : 1D array
        test input.
    Xtrain : 2D array
        training input matrix.
    model : TYPE
        regression model that implements .predict().
    N_samples_used : int, optional
        DESCRIPTION. The default is None.
    seed : int, optional
        random seed. The default is 1.
    reporting_percentage : int
        If this is 10, objective values are reported every 10 iterations. 
        The default is 0, which does not report on anything.

    Returns
    -------
    SV : 1D array
        Shapley value.

    '''
    rng = np.random.default_rng(seed)
    if (reporting_percentage > 100) | (reporting_percentage < 0):
        reporting_percentage = 0

    # Monte Carlo calculation of SV
    N,M = Xtrain.shape
    SV = np.zeros(M)
    z = np.zeros(M)

    if N_samples_used is None:
        N_samples_used = N
    if xtest.ndim == 2:
        xtest = xtest.ravel()

    deltaN = int(N_samples_used * reporting_percentage/100)
    for n in range(N_samples_used):
        if (reporting_percentage > 0) and ((n+1)%deltaN == 0):
            progress = int((n+1)/deltaN)
            print(f'{progress}..',end='')

        # Pick up one sample 
        n_idx = rng.choice(range(N))
        xn = Xtrain[n_idx,:]

        # Permute the indexes
        index_array = rng.permutation(M)

        # For each variable, find the corresponding S_i
        for i in range(M):
            i_loc = np.where(index_array==i)[0][0]
            S_i = index_array[0:i_loc]
            Sci = index_array[(i_loc+1):]

            z[S_i] = xtest[S_i]
            z[i]= xtest[i]
            z[Sci] = xn[Sci]
            f1 = model.predict(z.reshape(1,-1))

            z[i]= xn[i]
            f2 = model.predict(z.reshape(1,-1))
            SV[i] = SV[i] + (f1-f2)/N_samples_used
    print('..done')
    return SV



def LIME_deviation(xtest, ytest, model, N_grad=100, eta=0.01, l1=1e-5, seed=1):
    '''
    LIME for explaining deviations. (x^test, y^test) ---> grad
    Can take only a single point.
    This method can be used as the standard LIME with ytest =0.

    Parameters
    ----------
    xtest : 1D or 2D ndarray
        Input vector.
    ytest : float
        Output value.
    model : TYPE
        DESCRIPTION.
    N_grad : TYPE, optional
        DESCRIPTION. The default is 100.
    eta : float, optional
        DESCRIPTION. The default is 0.01.
    l1 : float, optional
        DESCRIPTION. The default is 1e-5.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    grad : 1D ndarray
        LIME explanation score (lasso regression coefficients).
    intercept : float
        Intercept value of the regression model.

    '''
    xtest = xtest.ravel()
    M = len(xtest)

    rng = np.random.default_rng(seed)
    Xlocal = rng.normal(loc=xtest,scale=eta,size=(N_grad,M))
    flocal = model.predict(Xlocal) - ytest

    fitLasso = linear_model.Lasso(alpha=l1,fit_intercept=True)
    fitLasso.fit(Xlocal,flocal)
    grad = fitLasso.coef_
    intercept = fitLasso.intercept_

    return grad, intercept

def LIME(xtest, model, N_grad=100, eta=0.01, l1=1e-5, seed=1):
    '''
    LIME for explaining deviations. (x^test, y^test) ---> grad
    Can take only a single point.
    This method can be used as the standard LIME with ytest =0.

    Parameters
    ----------
    xtest : 1D or 2D ndarray
        Input vector.
    model : TYPE
        DESCRIPTION.
    N_grad : TYPE, optional
        DESCRIPTION. The default is 100.
    eta : float, optional
        DESCRIPTION. The default is 0.01.
    l1 : float, optional
        DESCRIPTION. The default is 1e-5.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    grad : 1D ndarray
        LIME explanation score (lasso regression coefficients).
    intercept : float
        Intercept value of the regression model.

    '''

    return LIME_deviation(xtest=xtest, ytest=0, model=model, N_grad=N_grad,
                          eta=eta, l1=l1, seed=seed)

class Sinusoldal2D(BaseEstimator,RegressorMixin):
    '''
    Sinusoldal_2D benchmark model
    '''
    def __init__(self,a=1,b=1):
        '''
        Create Sinusoldal2D object: $f(x_1,x_2)=2 \cos(\pi a x)\cos(\pi b y)$

        Parameters
        ----------
        a : float, optional
            'a' parameter. The default is 1.
        b : float, optional
            'b' parameter. The default is 1.

        Returns
        -------
        None.

        '''
        self.a = a
        self.b = b

    def fit(self):
        '''
        This fit function is not used as there is no need to fit.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return self

    def predict(self,X_test):
        '''
        Predict function

        Parameters
        ----------
        X_test : 2D ndarray
            Data matrix where each row is a sample vector.

        Returns
        -------
        y_test : 1D ndarray
            Vectorized function for f(x_1, x_2)

        '''
        X_test=check_array(X_test)
        y_test = self.f(X_test[:,0], X_test[:,1])
        return y_test

    def f(self, x1, x2):
        '''
        Internal function used from predict()

        Parameters
        ----------
        x1 : TYPE
            DESCRIPTION.
        x2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return 2.*np.cos(self.a*np.pi*x1)*np.cos(self.b*np.pi*x2)



def consistency_analysis(df, reference_method, target_methods,
                         metric_names = ['tau','rho','smr','hit25']):
    '''
    Perform consistency analysis.

    Parameters
    ----------
    df : Pandas DataFrame of ndim=2
        Each row has a vector of attribution score plus 2 additional
        information, which are 'method' (GPA, LC, LIME, IG, SV, Z) and
        n_test (test sample index).
    reference_method : string
        The mane of the reference method against which the other methods are
        compared.
    target_methods : list-like
        List of the other methods compared.
    metric_names : TYPE, optional
        DESCRIPTION. The default is ['tau','rho','smr','hit25'].

    Returns
    -------
    df_means : pandas.DataFrame of ndim=2
        ([tau,rho,smr,hit25])x([LC,LIME,IG,SV,Z])-matrix.
    df_std : TYPE
        ([tau,rho,smr,hit25])x([LC,LIME,IG,SV,Z])-matrix..
    metric_names : TYPE
        [tau,rho,smr,hit25] by default

    '''

    n_tests = df['n_test'].unique()
    M = df.shape[1] - 2


    df_list = []
    for n_test in n_tests:

        # Choosing data
        #method_name= 'GPA'
        mask = (df['n_test'] == n_test) & (df['method'] == reference_method)
        GPA = df.loc[mask,:].iloc[0,0:M]
        scores = []

        for method_name in target_methods:
            mask = (df['n_test'] == n_test) & (df['method'] == method_name)
            score = df.loc[mask,:].iloc[0,0:M]

            # Computing the four statistics
            metrics = get_consistency_metrics(GPA, score)

            scores = scores + [pd.Series(metrics,index=metric_names)]

        # Converting to df
        df_score = pd.concat(scores,axis=1).T
        df_score['method'] = target_methods
        df_score['n_test'] = np.repeat(n_test,len(target_methods))
        df_list = df_list +[df_score]

    df_metric = pd.concat(df_list,ignore_index=True)

    # Computing statistics and package them
    means_list = []
    stdev_list = []
    for metric in metric_names:

        means = pd.Series(np.zeros([len(target_methods)]), index=target_methods)
        stddev = pd.Series(np.zeros([len(target_methods)]), index=target_methods)

        for method in target_methods:
            mask = (df_metric['method'] == method)
            data = df_metric[metric].loc[mask]
            means[method] = data.mean()
            stddev[method] = data.std()

        means_list = means_list + [means]
        stdev_list = stdev_list + [stddev]

    df_means = pd.concat(means_list,axis=1).T
    df_means.index = metric_names

    df_std = pd.concat(stdev_list,axis=1).T
    df_std.index = metric_names

    return df_means, df_std, metric_names


def get_consistency_metrics(x,y):
    '''
    Given two attribution score vectors, compute 4 consistensy metrics:
        - Kendall's tau
        - Spearman's rho
        - Sign matching ratio ("How are the signs consistent?")
        - hit at 25% ("Are the absolute top 25% members are the same?")

    Parameters
    ----------
    x : list-like
        One attribution score vector.
    y : list-like
        Another attribution score vector.

    Returns
    -------
    tau : float
        Kendall's tau.
    rho : float
        Spearman's rho.
    SMR : float
        Sign matching ratio.
    hit25 : float
        hit at 25%.

    '''

    x = np.array(list(x))
    y = np.array(list(y))

    absx = np.abs(x)
    absy = np.abs(y)

    N25 = int(len(absx)/4)

    x_binary = np.zeros(len(absx))
    x_binary[(x > 0)] = 1
    x_binary[(x < 0)] = -1

    y_binary = np.zeros(len(y))
    y_binary[(y > 0)] = 1
    y_binary[(y < 0)] = -1

    tau,p_tau = sp.stats.kendalltau(absx, absy)

    rho, p_rho = sp.stats.spearmanr(absx, absy)

    SMR = 1-(x_binary*y_binary <0).sum()/len(x_binary)

    #
    x_top25_set = set(np.argsort(-absx)[0:N25])
    y_top25_set = set(np.argsort(-absy)[0:N25])
    hit25 = 1-len(y_top25_set - x_top25_set)/N25

    return tau,rho,SMR, hit25

def anomaly_score(X,y,model,sigma_yf=1,Xtrain=None,ytrain=None):
    '''
    Computes anomaly score (outlier score) as negative log-likelihood of Gaussian
    predictive distribution. If sigma_yf=1, the score is basically the same
    as the squared error.

    Parameters
    ----------
    X : 2D numpy array
        Test predictor data matrix. Each row is a data entry (one sample).
    y : 1D numpy array
        Test target data
    model : scikit-learn-compatible object.
        Black-box prediction model f(x).
    sigma_yf : float, optional
        Predictive standard deviation of y, which can be estimated as the standard
        deviation of y-f(x), where f(x) is a regression function. The default is 1.
    Xtrain : 2D ndarray, optional
        Training dataset of predictors. Can be ignored if you provide sigma_yf.
         The default is None.
    ytrain : 1D ndarray, optional
        Training dataset of target variable. Can be ignored if you provide sigma_yf.
        The default is None.

    Returns
    -------
    a : 1D ndarray
        Anomaly score of each test samples.

    '''
    import numpy as np
    if (Xtrain is not None) and (ytrain is not None):
        sigma_yf =  (ytrain - model.predict(Xtrain)).std()

    a1 =  -0.5*np.log(2*np.pi) - 0.5*np.log(sigma_yf**2)
    a2 = -( (y.ravel() - model.predict(X))**2)/(2*(sigma_yf**2) )
    a= (-1)*(a1 + a2)
    return a

def show_scatter_simple(X_train,y_train,variable_names,
                        variable_idx_map,figsize=(9,7),ylabel='',
                        ylim=(0,5),xlim=None,alpha=0.3,marker_size=10,
                        label_font=14,tick_font=12):
    '''
    Given a collection of (x,y), draw y vs x_i scatter plots as specified by
    variable_idx_map. For example, variable_idx_map = np.array([[0,1,2,3,4],[5,6,7,8,9]]
    creates a 2x5 grid, where the first raw has 0-th, .., and 4-th variables.

    variable_names is used to annotate the x-axis


    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    variable_names : TYPE
        DESCRIPTION.
    variable_idx_map : TYPE
        DESCRIPTION.
    figsize : TYPE, optional
        DESCRIPTION. The default is (9,7).
    ylabel : TYPE, optional
        DESCRIPTION. The default is ''.
    ylim : TYPE, optional
        DESCRIPTION. The default is (0,5).
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.3.
    marker_size : TYPE, optional
        DESCRIPTION. The default is 10.
    label_font : TYPE, optional
        DESCRIPTION. The default is 14.
    tick_font : TYPE, optional
        DESCRIPTION. The default is 12.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''

    ix,iy=variable_idx_map.shape
    fig,ax = plt.subplots(ix,iy,figsize=(figsize[0],figsize[1]),sharey=True)

    for ii in range(ix):
        for jj in range(iy):
            idx = variable_idx_map[ii,jj]
            xx = X_train[:,idx]
            yy = y_train
            ax[ii,jj].scatter(x=xx,y=yy,c="black",alpha=alpha,s=marker_size)
            ax[ii,jj].set_xlabel(variable_names[idx],fontsize=label_font)
            ax[ii,jj].set_ylabel(ylabel,fontsize=label_font)
            if ylim is not None:
                ax[ii,jj].set_ylim(ylim[0],ylim[1])
            if xlim is not None:
                ax[ii,jj].set_xlim(xlim[0],xlim[1])
            ax[ii,jj].tick_params(axis='y',labelsize=tick_font)
            ax[ii,jj].tick_params(axis='x',labelsize=tick_font)

    fig.tight_layout()
    return fig,ax


def plot_scatter_selected(variable_to_plot, variable_names, X_train, y_train,
                          test_x1 = None, test_y1 = None,c1='red',mar1='s',s1=80,
                          test_x2 = None, test_y2 = None,c2='blue',mar2='^',s2=100,
                          ylim = None, xlim=None, figsize=(12,3), alpha=0.3,
                          markerSize=20,fontsize_xlabel = 14,fontsize_tick=12,
                          y_name='MEDV',fontsize_y=14):
    '''
    Draw pairwise scatter plots aligned horizontally.

    Parameters
    ----------
    variable_to_plot : array-like
        Variable names to plot. The entries must be in variable_names.
    variable_names : array-like
        The variable name corresponding to the columns of X_traing.
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    test_x1 : TYPE, optional
        DESCRIPTION. The default is None.
    test_y1 : TYPE, optional
        DESCRIPTION. The default is None.
    c1 : TYPE, optional
        DESCRIPTION. The default is 'red'.
    mar1 : TYPE, optional
        DESCRIPTION. The default is 's'.
    s1 : TYPE, optional
        DESCRIPTION. The default is 80.
    test_x2 : TYPE, optional
        DESCRIPTION. The default is None.
    test_y2 : TYPE, optional
        DESCRIPTION. The default is None.
    c2 : TYPE, optional
        DESCRIPTION. The default is 'blue'.
    mar2 : TYPE, optional
        DESCRIPTION. The default is '^'.
    s2 : TYPE, optional
        DESCRIPTION. The default is 100.
    ylim : TYPE, optional
        DESCRIPTION. The default is None.
    xlim : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (12,3).
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.3.
    markerSize : TYPE, optional
        DESCRIPTION. The default is 20.
    fontsize_xlabel : TYPE, optional
        DESCRIPTION. The default is 14.
    fontsize_tick : TYPE, optional
        DESCRIPTION. The default is 12.
    yname : string, optional
        The title of the y axis

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''
    import matplotlib.pyplot as plt
    import seaborn as sb; sb.set()
    import numpy as np
    fig,ax = plt.subplots(1,len(variable_to_plot),figsize=figsize,sharey=True)
    if type(variable_names) is list:
        variable_names = np.array(variable_names)
    if isinstance(test_x1,list):
        test_x1 = np.array(test_x1)
    elif isinstance(test_x1,np.ndarray):
        test_x1 = test_x1.ravel()

    if isinstance(test_x2,list):
        test_x2 = np.array(test_x2)
    elif isinstance(test_x2,np.ndarray):
            test_x2 = test_x2.ravel()

    for ii, variable in enumerate(variable_to_plot):
        idx = np.where(variable_names == variable)[0][0]

        if ii == 0:
            ax[ii].set_ylabel(y_name,fontsize=fontsize_y)
        ax[ii].scatter(x=X_train[:,idx],y=y_train,c="black",alpha=alpha,s=markerSize)
        ax[ii].set_xlabel(variable,fontsize=fontsize_xlabel)
        if ylim is not None:
            ax[ii].set_ylim(ylim[0],ylim[1])
        if xlim is not None:
            ax[ii].set_xlim(xlim[0],xlim[1])

        ax[ii].tick_params(axis='y',labelsize=fontsize_tick)
        ax[ii].tick_params(axis='x',labelsize=fontsize_tick)

        if test_x1 is not None:
            x,y = test_x1[idx], test_y1
            ax[ii].scatter(x=x,y=y,c=c1,marker=mar1,s=s1,edgecolors='white')
        if test_x2 is not None:
            x,y = test_x2[idx], test_y2
            ax[ii].scatter(x=x,y=y,c=c2,marker=mar2,s=s2,edgecolors='white')

    fig.tight_layout()
    return fig, ax

def plot_scatter_any_selected(variable_to_plot, variable_names, X_train, y_train,
                          test_X, test_y = None, c1='red',mar1='s',s1=80,
                          ylim = None, xlim=None, figsize=(12,3), alpha=0.3,
                          markerSize=20,fontsize_xlabel = 14,fontsize_tick=12,
                          y_name='MEDV',fontsize_y=14):
    '''
    Draw pairwise scatter plots aligned horizontally. Variable are chosen
    based on the name list provided by variable_names.
    If test_X and test_y are provided, the samples contained are plotted as
    highlighted points

    Parameters
    ----------
    variable_to_plot : TYPE
        DESCRIPTION.
    variable_names : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    test_X : TYPE, optional
        test point inputs of an arbitrary number. The default is None.
    test_y : TYPE, optional
        test point outputs of an arbitrary number. The default is None.
    c1 : TYPE, optional
        DESCRIPTION. The default is 'red'.
    mar1 : TYPE, optional
        DESCRIPTION. The default is 's'.
    s1 : TYPE, optional
        DESCRIPTION. The default is 80.
    c2 : TYPE, optional
        DESCRIPTION. The default is 'blue'.
    ylim : TYPE, optional
        DESCRIPTION. The default is None.
    xlim : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (12,3).
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.3.
    markerSize : TYPE, optional
        DESCRIPTION. The default is 20.
    fontsize_xlabel : TYPE, optional
        DESCRIPTION. The default is 14.
    fontsize_tick : TYPE, optional
        DESCRIPTION. The default is 12.
    yname : string, optional
        The title of the y axis

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''
    import matplotlib.pyplot as plt
    import seaborn as sb; sb.set()
    import numpy as np
    fig,ax = plt.subplots(1,len(variable_to_plot),figsize=figsize,sharey=True)
    if type(variable_names) is list:
        variable_names = np.array(variable_names)
    if isinstance(test_X,list):
        test_X = np.array(test_X)

    for ii, variable in enumerate(variable_to_plot):
        idx = np.where(variable_names == variable)[0][0]

        if ii == 0:
            ax[ii].set_ylabel(y_name,fontsize=fontsize_y)
        ax[ii].scatter(x=X_train[:,idx],y=y_train,c="black",alpha=alpha,s=markerSize)
        ax[ii].set_xlabel(variable,fontsize=fontsize_xlabel)
        if ylim is not None:
            ax[ii].set_ylim(ylim[0],ylim[1])
        if xlim is not None:
            ax[ii].set_xlim(xlim[0],xlim[1])

        ax[ii].tick_params(axis='y',labelsize=fontsize_tick)
        ax[ii].tick_params(axis='x',labelsize=fontsize_tick)

        if test_X is not None:
            for i_outlier in range(len(test_y)):
                x_outlier = test_X[i_outlier,idx]
                y_outlier = test_y[i_outlier]
                ax[ii].scatter(x=x_outlier, y=y_outlier,
                               c=c1, marker=mar1, s=s1, edgecolors='white')


    fig.tight_layout()
    return fig, ax


def plot_distributions(score_grid,score_center, df_dist,
                       method = '',
                       variable_to_plot=None, figsize = (3,6),
                       bw_ratio = 5, margin_ratio = 1.2,
                       font_title = 16, font_label = 14,
                       y_labelpad = 70, linewidth=4, show_y_label= True):
    '''
    Plot distribution + MAP estimation

    Parameters
    ----------
    score_grid : TYPE
        DESCRIPTION.
    score_center : TYPE
        DESCRIPTION.
    df_dist : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is ''.
    variable_to_plot : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (3,6).
    bw_ratio : TYPE, optional
        DESCRIPTION. The default is 5.
    margin_ratio : TYPE, optional
        DESCRIPTION. The default is 1.2.
    font_title : TYPE, optional
        DESCRIPTION. The default is 16.
    font_label : TYPE, optional
        DESCRIPTION. The default is 14.
    y_labelpad : TYPE, optional
        DESCRIPTION. The default is 70.
    linewidth : TYPE, optional
        DESCRIPTION. The default is 4.
    show_y_label : boolean
        If False, y label (variable name) is not drawn.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    '''

    variable_names = list(df_dist.columns)
    if variable_to_plot is None:
        variable_to_plot = variable_names.copy()

    fig, ax = plt.subplots(len(variable_to_plot),1,
                           figsize=figsize,sharey=True,sharex=True)

    x_min = score_grid[0]
    x_max = score_grid[-1]

    MAP_location = pd.Series(score_center,index=variable_names)

    for ii, vname in enumerate(variable_to_plot):

        ax[ii].set_xlim(x_min,x_max)

        ax[ii].tick_params(axis='y',labelleft=False)
        if ii == 0:
            ax[ii].set_title(method,fontsize = font_title)

        if show_y_label:
            ax[ii].set_ylabel(vname,fontsize=font_label,loc='bottom',
                             rotation=0, labelpad = y_labelpad)

        if vname == variable_to_plot[-1]:
            ax[ii].tick_params(axis='x',labelbottom=True)


        # "center" location
        x = MAP_location[vname]
        ax[ii].axvline(x,color='black',linewidth=linewidth)

        # Plotting distribution
        p = df_dist[vname]
        ax[ii].plot(score_grid,p,color='black')

    fig.tight_layout()
    return fig, ax
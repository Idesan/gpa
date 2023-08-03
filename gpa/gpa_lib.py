# -*- coding: utf-8 -*-
"""
Core library for GPA (Generative Perturbation Analysis)
with simplified comments for public release on Github.

Last updated on Feb 05, 2023

@author: Tsuyoshi Ide (ide@ide-research.net / tide@us.ibm.com )
"""
import numpy as np
import scipy as sp

def gpa_dist(delta_MAP,X,y,model,a,b,l2=0.1,l1_ratio=0,N_grid = 100,
                  grid_margin = 1.15):
    '''
    Estimates the distribution of the attribution score based on GPA.

    Parameters
    ----------
    delta_MAP : 1D ndarray
        MAP estimate of the attribution score, delta
    X : 2D ndarray
        Test data matrix whose rows are test sample vectors.
    y : 1D ndarray
        y-value corresponding to the rows of X.
    model : TYPE
        The black-box model that implements predict().
    a : float
        The a value coming from the Gamma prior. 2a corresponds to the degrees
        of freedom of the predictive t-distribution. 2a = N_test + 1 or 10 is
        recommended.
    b : float
        The b-value. Typically set to a*(sigma_yf**2)/c_b, where c_b is 1 or 10.
    l2 : TYPE, optional
        DESCRIPTION. The default is 0.1.
    l1_ratio : TYPE, optional
        DESCRIPTION. The default is 0.
    N_grid : TYPE, optional
        DESCRIPTION. The default is 100.
    grid_margin: float, optional
        The default is 1.15. If delta's max is 1, then the distribution will be
        computed within the range of [-1.15, 1.15].

    Returns
    -------
    prob_dist : 1D ndarray
        probability distribution that sums to 1.
    score_grid : 1D ndarray
        The grid point locations at which the probability is computed.

    '''

    if X.ndim == 1:
        X = X.reshape(1,-1)
        b = np.array([b])

    N_sample,M = X.shape

    if isinstance(b,(int,float)):
        b = np.repeat(b,N_sample)

    # The constant term of the log likelihood
    C2 = -(N_sample/2.)*np.log(2*np.pi) - 0.5*np.log(b).sum() \
        +N_sample*np.log(sp.special.gamma(a+0.5)/sp.special.gamma(a)) \
            + 0.5*M*np.log(l2) - (M/2.)*np.log(2*np.pi)

    # For the grid points on which probabilities are computed. 
    score_max = np.max(np.abs(delta_MAP))*grid_margin
    score_min = np.max(np.abs(delta_MAP))*(-grid_margin)

    score_grid = np.linspace(score_min, score_max,num=N_grid)

    huge_X = np.tile(X, [N_grid*M,1])
    huge_delta= np.tile(delta_MAP.reshape([1,-1]),[N_sample*N_grid*M,1])

    huge_y = np.tile(y.reshape([-1,1]),[N_grid*M,1])
    huge_b = np.tile(b.reshape([-1,1]),[N_grid*M,1])

    # Vectorized data structure for faster computation. 
    big_score_grid = np.tile(score_grid.reshape([1,-1]),[N_sample,1])
    big_score_grid = big_score_grid.flatten(order='F')

    # Creating "huge_delta", stacking M matrices of (N_sample*N_grid)xM from the top to the bottom.
    for idx in range(M):
        idx_start = idx*N_sample*N_grid
        idx_end = idx_start + N_sample*N_grid

        huge_delta[idx_start:idx_end,idx] = big_score_grid[:]

    # Evaluating the black-box function
    huge_f = model.predict(huge_X + huge_delta)
    huge_ln = np.log(1 + (huge_y.ravel() - huge_f)**2/(2*huge_b.ravel()))

    # Computing the probability from a vector of N_sample*N_grid*M entries.
    prob_dist = np.empty([M,N_grid])
    big_ln = np.empty([N_sample*N_grid])
    for idx in range(M):
        idx_start = idx*N_sample*N_grid
        idx_end = idx_start + N_sample*N_grid

        big_ln[:] = huge_ln[idx_start:idx_end]

        log_term = big_ln.reshape([N_sample,N_grid],order='F').sum(axis=0)
        reg_term = (-0.5)*l2*(score_grid**2) + C2 - l2*l1_ratio*np.abs(score_grid)
        wa = reg_term - (a+0.5)*log_term

        # Normalization condition.
        prob_dist[idx,:] = np.exp(wa) / np.exp(wa).sum()

    return prob_dist,score_grid


def gpa_map(X, y, model, a, b, l2=0.1, l1_ratio=0.5,
        lr = 0.05, lr_shrinkage = 0.9,
        itr_max=50, RAE_th=1e-3, reporting_interval='auto',
        N_grad=10, eta_stddev = 0.1, delta_init_scale = 1e-8,
        seed_initialize=1,
        seed_grad = 1, h_minimum=1.e-8, verbose=False):
    '''
    Finding a MAP solution for Generative Perturbation Analysis (GPA).

    Parameters
    ----------
    X : 2D ndarray
        Row-based data matrix for the input variables. Typically, X has only
        one row. Multiple rows are for collective attribution.
    y : list-like
        y values corresponding to the rows of X
    model : TYPE
        black-box regression function object with a predict() implemented.
    a : float
        The shape parameter of the gamma prior. 2a corresponds to 
        the degrees of freedom of the resulting t-distribution.
    b : float
        The rate parameter of the gamma prior. Can be an array rather than a scalar. 
    l2 : float, optional
        The precision parameter of the Gaussian prior. The default is 0.1.
    l1_ratio : float, optional
        The L1 strength of the elastic net prior, defined relative to the l2 strength. 
        l2*l1_ratio gives the l1 regularization strength. The default is 0.5.
    lr : float, optional
        The learning rate kappa. The default is 0.05.
    lr_shrinkage : float, optional
        The shrinkage rate of the learning rate (see Ide et al AAAI 21). 
        The default is 0.9.
    itr_max : int, optional
        The maximum number of iteration. The default is 50.
    RAE_th : float, optional
        The threshold for the relative absolute error. Used to judge the convergence. 
        The default is 1e-3.
    reporting_interval : int, optional
        If this is 10, errors are reported every 10 iterations. The default is 'auto',
        which uses reporting_interval = int(itr_max/10). 
    N_grad : int, optional
        The number of perturbations for gradient estimation. The default is 10.
    eta_stddev : float, optional
        Standard deviation of Gaussian used for Monte Carlo gradient estimation.
        The default is 0.1.
    delta_init_scale : float, optional
        The scale of random small noise for initializing the MAP value of delta 
        The default is 1e-8.
    seed_initialize : int, optional
        Random seed for initializing delta. The default is 1.
    seed_grad : int, optional
        Random seed for Monte Carlo gradient estimation. The default is 1.
    h_minimum : float, optional
        Minimum threshold of the absolute increment to avoid divide-by-zero. 
        Generated increments below this threshold will be discarded. The default is 1.e-8.
    verbose : boolean, optional
        Set True if you want detailed updates in the course of iteration. The default is False.

    Returns
    -------
    delta : array
        The MAP estimate of the attribution score delta.
    params : dict
        gradients : array
            Each row is the gradient for each test sample.
        obj_values : float
            The objective value (log likelihood)
        itr : int
            The number of iteration upon finishing the loop. 
        lr_final : float
            The final value of the learning rate. Note that learning rate gets smaller geometrically. 
        RAEs_delta : 
            Relative absolute error of delta of each iteration
        RAEs_objec : 
            RAE of the objective value of delta of each iteration

    '''

    if reporting_interval == 'auto':
        reporting_interval = int(itr_max/10)
    if X.ndim == 1:
        X= X.reshape(1,-1)
        y= np.array([y])

    N_test,M = X.shape

    # Constant term of the log likelihood ##### J(delta)-dependent #####
    gam_ratio = np.log( sp.special.gamma(a+0.5)/sp.special.gamma(a) )
    const_obj = 0.5*N_test*np.log(2*np.pi) \
        + 0.5*(np.log(b)).sum() + N_test*gam_ratio \
            -0.5*M*np.log(l2) +0.5*M*np.log(2.*np.pi)

    # Initializing delta -----------------------------------
    rng = np.random.default_rng(seed_initialize)
    delta_initial = rng.normal(0,delta_init_scale,M)
    delta_l1_initial = np.max([np.abs(delta_initial).sum(),delta_init_scale])
    delta = delta_initial.copy()

    delta_old = np.repeat(np.Inf, M)
    objective_old = -np.Inf

    # Allocating memory
    gradients = np.empty([N_test,M]) # local gradient in each row
    g_vec = np.empty(M) # The g vector of the prox grad
    Z = np.empty(X.shape) # The data matrix that keeps getting updated. 
    DeltaN = np.empty(N_test)

    # Iterative updates start -----------------------
    obj_values = []
    RAEs_delta = []
    RAEs_objec = []

    for itr in range(itr_max):

        Z[:,:] = X + delta.reshape(1,-1)

        # Computing y - f(x+delta) 
        DeltaN[:] = y - model.predict(Z)

        # Computing local gradients (stored in each row)
        gradients[:,:] = local_gradient_vec2(Z=Z, model=model, N_grad=N_grad,
                                             eta_stddev=eta_stddev,
                                             seed = seed_grad,
                                             h_minimum=h_minimum)

        # Computing the g vector. This is a 1D array. ##### J(delta)-dependent #####
        g_vec[:] = ( gradients*( (DeltaN/(2*b + DeltaN**2)\
                                 ).reshape(-1,1)) ).sum(axis=0)
        g_vec[:] = (1-lr*l2)*delta + lr*(2*a+1)*g_vec

        # The lasso solution with the L1 term (proximal gradient algorithm)
        delta[:] = prox_l1(g_vec,lr*l2*l1_ratio)

        # Computing the objective function ##### J(delta)-dependent #####
        obj_value = (a+0.5)*(np.log(1 + (DeltaN**2/(2*b))) ).sum() \
            + 0.5*l2*(delta**2).sum() + const_obj
        obj_values = obj_values + [obj_value]

        # Checking convergence
        delta_L1norm = np.max([np.abs(delta).sum(),delta_l1_initial])
        RAE_delta = np.abs(delta - delta_old).sum()/delta_L1norm
        RAEs_delta = RAEs_delta + [RAE_delta]

        RAE_objec = np.abs(obj_value - objective_old)/np.abs(obj_value)
        RAEs_objec = RAEs_objec + [RAE_objec]

        if (RAE_delta  <= RAE_th) and (RAE_objec <= RAE_th):
            break

        # Prepping for the next round
        delta_old[:] = delta[:]
        objective_old = obj_value
        lr = lr*lr_shrinkage

        # Reporting
        if ((itr+1)%reporting_interval == 0) and verbose:
            print(f'{itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
            print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    print(f'finished:itr={itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
    print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    params = {'gradients':gradients, 'obj_values':np.array(obj_values),
              'itr':(itr+1), 'lr_final':lr,
              'RAEs_delta':RAEs_delta, 'RAEs_objec':RAEs_objec}
    return delta, params


def gpa_map_gaussian(X, y, model, stddev_yf, l2=0.1, l1_ratio=0.5,
        lr = 0.05, lr_shrinkage = 0.9,
        itr_max=50, RAE_th=1e-3, reporting_interval='auto',
        N_grad=10, eta_stddev = 0.1, delta_init_scale = 1e-8, seed_initialize=1,
        seed_grad = 1, h_minimum=1.e-8, verbose=False):
    '''
    Computes the MAP value of delta based on (Gaussian observation)
    +(elastic net prior) model. This should return the same score as
    Likelihood Compensation (LC; Ide et al. AAAI 21).

    Uses local_gradient_vec2() for gradient estimation.

    Parameters
    ----------
    X : array-like
        Test data matrix of test input. If N_test = 1, X becomes 1-row matrix.
    y : array
        Test data output.
    model : TYPE
        Black-box regression model object that allows .predict().
    stddev_yf : float
        Predictive standard deviation. This is NOT the variance. 
    l2 : float, optional
        L2 regularization strength or the precision of the Gaussian prior。The default is 0.1.
    l1_ratio : TYPE, optional
        L1 regularization strength relative to that of L2. The default is 0.5.
    lr : float, optional
        Learning rate kappa. The default is 0.05.
    lr_shrinkage : float, optional
        lr gets shrunken by lr = lr*lr_shrinkage. The default is 0.9.
    itr_max : int, optional
        Maximum number of iterations. The default is 50.
    RAE_th : float, optional
        Threshold of convergence of the relative absolute error of delta and 
        the negative log likelihood. The default is 1e-3.
    reporting_interval : int, optional
        How often you want to get updates. The default is 'auto', which uses
        reporting_interval = int(itr_max/10).
    N_grad : int, optional
        The number of random perturbations for local gradient estimation. The default is 10.
    eta_stddev : TYPE, optional
        The standard deviation used in Monte Carlo gradient estimation. The default is 0.1.
    delta_init_scale : TYPE, optional
        The scale of delta for random initialization. The default is 1e-8.
    seed_initialize : int, optional
        Random seed of initialization of delta. The default is 1.
    seed_grad : TYPE, optional
        Random seed of random perturbation in gradient estimation. The default is 1.
    h_minimum : TYPE, optional
        The minimum threshold of absolute perturbation, below which the perturbation is discarded. 
        This is to avoid divide-by-zero. The default is 1.e-8.
    verbose : boolean, optional
        If True, print errors. The default is False.

    Returns
    -------
    delta : ndarray
        The value of delta (LC).
    params : dict
        gradients : array
            Each row is the gradient for each test sample.
        obj_values : float
            The objective value (log likelihood)
        itr : int
            The number of iteration upon finishing the loop. 
        lr_final : float
            The final value of the learning rate. Note that learning rate gets smaller geometrically. 
        RAEs_delta : 
            Relative absolute error of delta of each iteration
        RAEs_objec : 
            RAE of the objective value of delta of each iteration

    '''

    if reporting_interval == 'auto':
        reporting_interval = int(itr_max/10)
    if X.ndim == 1:
        X= X.reshape(1,-1)
        y= np.array([y])

    # Verifying input
    N_test,M = X.shape
    if type(stddev_yf) is not np.ndarray:
        local_lambda = 1/np.repeat(stddev_yf**2, N_test)

    # Constant term of the negative log likelihood to be minimized. ##### J(delta)-dependent #####
    const_obj = 0.5*M*np.log(2.*np.pi) -0.5*M*np.log(l2) \
        +0.5*N_test*np.log(2*np.pi) - 0.5*np.log(local_lambda).sum()

    # Initializing delta -----------------------------------
    rng = np.random.default_rng(seed_initialize)
    delta_initial = rng.normal(0,delta_init_scale,M)
    delta_l1_initial = np.max([np.abs(delta_initial).sum(),delta_init_scale])
    delta = delta_initial.copy()

    delta_old = np.repeat(np.Inf, M)
    objective_old = -np.Inf

    # Assigning memory space
    gradients = np.empty([N_test,M]) # local gradient in the row.
    g_vec = np.empty(M) # The g vector of prox grad 
    Z = np.empty(X.shape) # The data matrix that keeps getting updated
    DeltaN = np.empty(N_test)

    # Iteration starts -----------------------------
    obj_values = []
    RAEs_delta = []
    RAEs_objec = []

    for itr in range(itr_max):

        Z[:,:] = X + delta.reshape(1,-1)

        # Computing y - f(x+delta) 
        DeltaN[:] = y - model.predict(Z)

        # Computing the gradient that is stored in the rows for each test sample. 
        gradients[:,:] = local_gradient_vec2(Z=Z, model=model, N_grad=N_grad,
                                             eta_stddev=eta_stddev,
                                             seed= seed_grad, h_minimum=h_minimum)

        # Computing the g vector (1D array) ##### J(delta)-dependent #####
        g_vec[:] = (gradients*((DeltaN*local_lambda).reshape(-1,1))).sum(axis=0)
        g_vec[:] = (1 - lr*l2)*delta + lr*g_vec

        # Prox grad with L1 regularization
        delta[:] = prox_l1(g_vec,lr*l2*l1_ratio)

        # Computing the objective function ##### J(delta)-dependent #####
        obj_value = 0.5*((DeltaN**2)*local_lambda).sum() \
            + 0.5*l2*(delta**2).sum() + const_obj
        obj_values = obj_values + [obj_value]

        # Checking convergence
        delta_L1norm = np.max([np.abs(delta).sum(),delta_l1_initial])
        RAE_delta = np.abs(delta - delta_old).sum()/delta_L1norm
        RAEs_delta = RAEs_delta + [RAE_delta]

        RAE_objec = np.abs(obj_value - objective_old)/np.abs(obj_value)
        RAEs_objec = RAEs_objec + [RAE_objec]

        if (RAE_delta  <= RAE_th) and (RAE_objec <= RAE_th):
            break

        # Prepping for the next round
        delta_old[:] = delta[:]
        objective_old = obj_value
        lr = lr*lr_shrinkage

        # Reporting
        if ((itr+1)%reporting_interval == 0) and verbose:
            print(f'{itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
            print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    print(f'finished:itr={itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
    print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    params = {'gradients':gradients, 'obj_values':np.array(obj_values),
              'itr':(itr+1), 'lr_final':lr,
              'RAEs_delta':RAEs_delta, 'RAEs_objec':RAEs_objec}

    return delta, params



def local_gradient_vec2(Z, model,N_grad, eta_stddev=0.1, seed=1,h_minimum=1.e-8):
    '''
    Given a black-box function f(x), simultaneously computes the local gradient
    at each of the rows of the data matrix Z.

    If you have 5 samples of 10-dimensional vectors, you will get a gradient
    matrix of the same size: 5x10 matrix, where each row is the gradient
    at the corresponding sample.

    Note that `model` is a black-box function. Unlike autograd functions in
    deep learning frameworks, we do not need any analytic form of the model.

    Parameters
    ----------
    Z : 2D ndarray of float
        data matrix whose rows are the coordinates at which gradient is evaluated.
    model : TYPE
        An object representing a black-box function. `predict()` has to be available.
    N_grad : int
        The number of perturbations generated. Typically, this is 10.
    eta_stddev : float, optional
        Standard deviation of the perturbations. The default is 0.1.
    seed : int, optional
        Seed for random perturbations. The default is 1.
    h_minimum : float, optional
        Allowable smallest scale of the perturbation. Note that the actual
        minimum threshold is given by h_minimum*eta_stddev.
        The default is 1.e-8.

    Returns
    -------
    grad_matrix : 2D ndarray
        Matrix of the gradients in the rows.

    '''

    N_test, M = Z.shape
    grad_matrix = np.zeros([N_test,M])

    # hh ベクトルを作る
    h0 = np.random.default_rng(seed).normal(0,eta_stddev,N_grad)

    h = h0[abs(h0) >= h_minimum*eta_stddev] # Discard too small perturbations.
    NN_grad = len(h) ## Use this instead of N_grad. If there is any discarded perturbation, 
    # NN_grad will be smaller than the original N_grad. 
    hh = np.zeros([N_test,NN_grad]) + h
    hh = hh.flatten(order='F')

    # Creating the long hhh vector to leverage the vectorized computation capability of numpy. 
    hhh = np.tile(hh.reshape(-1,1),[M,1])

    # Creating huge data matrix
    ZB0 = np.tile(Z,[NN_grad*M,1])
    ZB = ZB0.copy()

    # Evaluate the function value. Just one time. 
    ff0 = model.predict(ZB0)

    # Concatenating perturbed data matrices. Further concatenate them M times. 
    # Then fill values to them. The only difference is that the idx-th column is perturbed. 
    for idx in range(M):
        n_start = N_test*NN_grad*idx
        n_end = n_start + N_test*NN_grad
        ZB[n_start:n_end,idx] = ZB0[n_start:n_end,idx] + hh

    # Evaluate the function value. Just one time. 
    ff = model.predict(ZB)
    df = (ff-ff0)/hhh.ravel()

    for idx in range(M):
        n_start = N_test*NN_grad*idx
        n_end = n_start + N_test*NN_grad
        ge =  df[n_start:n_end].reshape([N_test,N_grad],order='F')
        grad_matrix[:,idx] =ge.mean(axis=1)

    return grad_matrix


def prox_l1(phi,mu):
    '''
    Proximal operator for L1 regularization.
    prox(phi | mu||phi||)

    Parameters
    ----------
    phi : 1D ndarray
        Input argument of the L1 prox operator.
    mu : float
        L1 regularization strength. Note: when used in the proximal gradient
        method, you need to multiply the l1 strength by the learning rate.

    Returns
    -------
    phi : 1D ndarray
        Regularized solution, i.e., prox(phi | mu||phi||).

    '''
    mask1 = (abs(phi) <= mu)
    mask2 = (phi < (-1)*mu)
    mask3 = (phi > mu)
    phi[mask1] = 0
    phi[mask2] = phi[mask2] + mu
    phi[mask3] = phi[mask3] - mu
    return phi

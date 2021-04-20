from numpy import linalg, zeros, ones, hstack, asarray
import itertools


def basis_vector(n, i):
    """ Return an array like [0, 0, ..., 1, ..., 0, 0]
    >>> from multipolyfit.core import basis_vector
    >>> basis_vector(3, 1)
    array([0, 1, 0])
    >>> basis_vector(5, 4)
    array([0, 0, 0, 0, 1])
    """
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    """ Turns a row vector into a column vector """
    return x.reshape(x.shape + (1,))

def multipolyfit(xs, y, deg, full=False, model_out=True, powers_out=False):
    """
    Least squares multivariate polynomial fit
    Fit a polynomial like ``y = a**2 + 3a - 2ab + 4b**2 - 1``
    with many covariates a, b, c, ...
    Parameters
    ----------
    xs : array_like, shape (M, k)
         x-coordinates of the k covariates over the M sample points
    y :  array_like, shape(M,)
         y-coordinates of the sample points.
    deg : int
         Degree o fthe fitting polynomial
    model_out : bool (defaults to True)
         If True return a callable function
         If False return an array of coefficients
    powers_out : bool (defaults to False)
         Returns the meaning of each of the coefficients in the form of an
         iterator that gives the powers over the inputs and 1
         For example if xs corresponds to the covariates a,b,c then the array
         [1, 2, 1, 0] corresponds to 1**1 * a**2 * b**1 * c**0
    See Also
    --------
        numpy.polyfit
    """
    y = asarray(y).squeeze()
    rows = y.shape[0]
    xs = asarray(xs)
    num_covariates = xs.shape[1]
    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))

    generators = [basis_vector(num_covariates+1, i)
                            for i in range(num_covariates+1)]

    # All combinations of degrees
    powers = list(map(sum, itertools.combinations_with_replacement(generators, deg)))


    # Raise data to specified degree pattern, stack in order
    A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))

    beta = linalg.lstsq(A, y, rcond=None)[0]
    if model_out:
        return mk_model(beta, powers)

    if powers_out:
        return beta, powers
    return beta

def mk_model(beta, powers):
    """ Create a callable pyTaun function out of beta/powers from multipolyfit
    This function is callable from within multipolyfit using the model_out flag
    """

    # Create a function that takes in many x values
    # and returns an approximate y value
    def model(*args):


        num_covariates = len(powers[0]) - 1
        args=args[0]
        if len(args)!=(num_covariates):
            raise ValueError("Expected %d inputs"%num_covariates)
        xs = [1]+args
        return sum([coeff * (xs**p).prod()
                             for p, coeff in zip(powers, beta)])
    return model

def mk_sympy_function(beta, powers):
    from sympy import symbols, Add, Mul, S

    num_covariates = len(powers[0]) - 1
    xs = (S.One,) + symbols('x0:%d'%num_covariates)
    return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)])
                        for power, coeff in zip(powers, beta)])

##################################################################

import numpy as np
import matplotlib.pyplot as plt

## payoffs of a vanilla call and put
def payoff_call(S, K):
    return max(sum(S)/len(S) - K, 0)

def payoff_put(S, K):
    return max(K - sum(S)/len(S), 0)

## paths generator for Monte Carlo calculation
def LogNormalPathsGenerator(M, N, T, drift, Sig, S0):
    d = len(S0)
    paths=[]
    s = [0]*d
    for i in range(d):
        s[i]=np.ones(M)*S0[i]
    paths.append([S0 for i in range(M)])
    dt = T/N
    sigma = np.zeros(d)
    for i in range(d):
        sigma[i] = np.sqrt(np.sum(Sig[i,:]**2))
    for k in range(1,N+1):
        path =[]
        W = [0]*len(S0)
        for i in range(d):
            W[i] = np.random.normal(0,1,M)*dt**.5
        W = np.array(W)
        dx = np.dot(Sig, W)
        for i in range(d):
            ds = drift*dt*s[i]+dx[i]*s[i]
            s[i] += ds
            #s[i] = S0[i]*np.exp((drift-sigma[i]**2/2)*dt+dx[i])
        for j in range(M):
            path.append([s[i][j] for i in range(d)])
        paths.append(path)
    return paths

r = .1
sigma = .4
S0 = [50,40]
#Sig = np.eye(len(S0))*sigma
Sig = np.array([[.4,0],[0,.3]])
K = 40
T = 1

M = 1000
N = 10

def discountedPayoffAtn(payoff, n, r, S, K, T, N):
    return np.exp(-r*n*T/N)*payoff(S, K)

def LeastSquaresMC(paths, T, r, K, payoff, deg):
    M, N =  len(paths[0]), len(paths)
    N -=1
    tau = np.full((M,N+1), N)

    j=N-1
    while j>=1:
        X = np.array(paths[j])

        Stau = [paths[tau[m,j+1]][m] for m in range(M)]
        Y=[]
        for m in range(M):
            Y.append(discountedPayoffAtn(payoff, tau[m,j+1], r, Stau[m], K, T, N))
        Y= np.array(Y)
        reg = multipolyfit(X, Y, deg)

        for m in range(M):
            phi = discountedPayoffAtn(payoff, j, r, paths[j][m], K, T, N)
            if phi >= reg(paths[j][m]):
                tau[m,j]=j
            else:
                tau[m,j]=tau[m,j+1]
        j-=1

    Stau1 = [paths[tau[m,1]][m] for m in range(M)]
    Y=[]
    for m in range(M):
        Y.append(discountedPayoffAtn(payoff, tau[m,1], r, Stau1[m], K, T, N))
    Y = np.array(Y)

    return max(payoff(paths[0][0],K) ,np.mean(Y))

means = []
plt.title("bermudan put value variation with respect to the number of trajectories simulated")

MM = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
for M in MM:
    put = []
    call = []
    for i in range(10):
        paths = LogNormalPathsGenerator(M, N, T, r, Sig, S0)
        put.append(LeastSquaresMC(paths, T, r, K, payoff_put, 6))
        call.append(LeastSquaresMC(paths, T, r, K, payoff_call, 7))
    x = [M]*10
    plt.plot(x, put, color='black')
    plt.scatter(x[0], np.mean(np.array(put)), color ="blue")
    means.append(np.mean(np.array(put)))

    print(M/1000)

plt.plot(MM, means, color="blue")
plt.xlabel("number of trajectories simulated")
plt.ylabel("bermudan put value")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense

import time

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
    self.paths.append([S0 for i in range(M)])
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

def NN_approximation(j, tau, paths, payoff):
    M, N =  len(paths[0]), len(paths)
    d = len(paths[0][0])
    model = Sequential()
    model.add(Dense(3*d, input_dim=1, kernel_initializer='he_normal', activation="elu"))

    #hidden layers
    model.add(Dense(4*d, activation="elu"))

    #hidden layers
    model.add(Dense(3*d, activation="elu"))

    #output layer
    model.add(Dense(1,activation="elu"))

    model.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy','mse', 'mae', 'mape'])

    X = np.array(paths[j])

    Stau = [paths[tau[m,j+1]][m] for m in range(M)]
    Y=[]
    for m in range(M):
        Y.append(discountedPayoffAtn(payoff, tau[m,j+1], r, Stau[m], K, T, N))
    Y= np.array(Y)

    history = model.fit(X, Y, batch_size=10, epochs=200, verbose=0)

    return (model,history)

r = .1
sigma = .4
S0 = [50]
Sig = np.eye(len(S0))*sigma
K = 40
T = 1

M = 1000
N = 100

generatedPaths = LogNormalPathsGenerator(M, N, T, r, sigma, S0)
#generatedPaths.plot()
paths = generatedPaths.paths
#print(paths)

def discountedPayoffAtn(payoff, n, r, X, K, T, N):
    return np.exp(-r*n*T/N)*payoff(X, K)

HISTORY=[]
HISTORY2=[]

def DNN_pricing(paths, T, r, K, payoff):
    M, N =  len(paths[0]), len(paths)
    N -=1
    tau = np.full((M,N+1), N)
    j=N-1
    while j>=1:
        model,history = NN_approximation(j,tau,paths,payoff)
        HISTORY.append(history.history["loss"])
        HISTORY2.append(history)
        for m in range(M):
            phi = discountedPayoffAtn(payoff, j, r, paths[m,j], K, T, N)
            if phi >= model.predict(np.array(paths[j][m])):
                tau[m,j]=j
            else:
                tau[m,j]=tau[m,j+1]
        j-=1
        print(j)
    Stau1 = np.array([paths[m,tau[m,1]] for m in range(M)])
    return max(payoff(paths[0,0],K),np.mean(discountedPayoffAtn(payoff, tau[:,1], r, Stau1, K, T, N)))

t0 = time.time()
print("american put value: ",DNN_pricing(paths, T, r, K, payoff_put))
t1 = time.time()
print(t1-t0)
print("american call value: ",DNN_pricing(paths, T, r, K, payoff_call))
print(time.time()-t1)

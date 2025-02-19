import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense

import time

## payoffs of a vanilla call and put
def payoff_call(S, K):
    return np.maximum(S - K, 0)

def payoff_put(S, K):
    return np.maximum(K - S, 0)

class LogNormalPathsGenerator:
    def __init__(self, M, N, T, drift, Sig, S0):
        self.paths=[]
        s=np.ones(M)*S0
        self.paths.append(np.copy(s))
        dt = T/N
        for k in range(1,N+1):
            dx = np.random.normal(0,1,M)*dt**.5
            ds = drift*dt*s+sigma*dx*s
            s += ds
            #s = s*np.exp((drift-sigma**2/2)*dt+sigma*dx)
            self.paths.append(np.copy(s))
        self.paths = np.transpose(np.array(self.paths))

    def plot(self):
        plt.title("Lognormal generated paths")
        t = np.linspace(0,N,N+1)
        for k in range(M):
            plt.plot(t, self.paths[k,:])
        plt.xlabel("time")
        plt.ylabel("stock price")
        plt.show()

def approximation(j, tau, paths, payoff):
    M, N = paths.shape
    model = Sequential()
    model.add(Dense(3, input_dim=1, kernel_initializer='he_normal', activation="elu"))

    #hidden layers
    model.add(Dense(4, activation="elu"))

    #hidden layers
    model.add(Dense(3, activation="elu"))

    #output layer
    model.add(Dense(1,activation="elu"))

    model.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy','mse', 'mae', 'mape'])

    Stau = np.array([paths[m,tau[m,j+1]] for m in range(M)])
    Y = discountedPayoffAtn(payoff, tau[:,j+1], r, Stau, K, T, N)

    X = paths[:,j]

    history = model.fit(X, Y, batch_size=10, epochs=200, verbose=0)

    return (model,history)

r = .1
sigma = .4
S0 = 50
K = 40
T = 1

N = 3

def discountedPayoffAtn(payoff, n, r, X, K, T, N):
    return np.exp(-r*n*T/N)*payoff(X, K)


def DNN_pricing(paths, T, r, K, payoff):
    M, N = paths.shape
    N -=1
    tau = np.full((M,N+1), N)
    j=N-1
    while j>=1:
        print(j)
        model,history = approximation(j,tau,paths,payoff)
        for m in range(M):
            phi = discountedPayoffAtn(payoff, j, r, paths[m,j], K, T, N)
            if phi >= model.predict(np.array([paths[m,j]])):
                tau[m,j]=j
            else:
                tau[m,j]=tau[m,j+1]
        j-=1
    Stau1 = np.array([paths[m,tau[m,1]] for m in range(M)])
    return max(payoff(paths[0,0],K),np.mean(discountedPayoffAtn(payoff, tau[:,1], r, Stau1, K, T, N)))

means = []
plt.title("bermudan put value variation with respect to the number of trajectories simulated (neural network)")

MM = [i*100 for i in range(1,10)]
for M in MM:
    put = []
    call = []
    for i in range(10):
        paths = LogNormalPathsGenerator(M, N, T, r, sigma, S0).paths
        put.append(DNN_pricing(paths, T, r, K, payoff_put))
        #call.append(DNN_pricing(paths, T, r, K, payoff_call))
    x = [M]*10
    plt.plot(x, put, color='black')
    plt.scatter(x[0], np.mean(np.array(put)), color ="cyan")
    means.append(np.mean(np.array(put)))
    print(M/100)


plt.plot(MM, means, color="blue")
plt.xlabel("number of trajectories simulated")
plt.ylabel("bermudan put value")
plt.show()

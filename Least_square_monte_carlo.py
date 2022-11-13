import numpy as np
import matplotlib.pyplot as plt

## payoffs of a vanilla call and put
def payoff_call(S, K):
    return np.maximum(S - K, 0)

def payoff_put(S, K):
    return np.maximum(K - S, 0)

class LogNormalPathsGenerator:
    def __init__(self, M, N, T, drift, sigma, S0):
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

r = .1
sigma = .4
S0 =50
K = 40
T = 1

M = 1000
N = 100

generatedPaths = LogNormalPathsGenerator(M, N, T, r, sigma, S0)
generatedPaths.plot()
paths = generatedPaths.paths


def discountedPayoffAtn(payoff, n, r, S, K, T, N):
    return np.exp(-r*n*T/N)*payoff(S, K)

def LeastSquaresMC(paths, T, r, K, payoff, deg):
    M, N = paths.shape
    N -=1
    tau = np.full((M,N+1), N)
    j=N-1
    while j>=1:
        X = paths[:,j]
        Stau = np.array([paths[m,tau[m,j+1]] for m in range(M)])
        Y = discountedPayoffAtn(payoff, tau[:,j+1], r, Stau, K, T, N)
        coeffs = np.polynomial.polynomial.polyfit(X, Y, deg)
        for m in range(M):
            phi = discountedPayoffAtn(payoff, j, r, paths[m,j], K, T, N)
            if phi >= np.polynomial.polynomial.polyval(paths[m,j], coeffs):
                tau[m,j]=j
            else:
                tau[m,j]=tau[m,j+1]
        j-=1
    Stau1 = np.array([paths[m,tau[m,1]] for m in range(M)])
    return max(payoff(paths[0,0],K) ,np.mean(discountedPayoffAtn(payoff, tau[:,1], r, Stau1, K, T, N)))

print("american put value: ",LeastSquaresMC(paths, T, r, K, payoff_put, 20))
print("american call value: ",LeastSquaresMC(paths, T, r, K, payoff_call, 10))

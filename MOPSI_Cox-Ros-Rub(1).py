import math
import matplotlib.pyplot as plt
import webbrowser

r = 0.1           # risk free rate
# disc = 1/(1+r)    # discount rate
disc = math.exp(-r)   # continous discount rate
sigma = 0.4       # volatility
q = 0             # dividend rate    # note that the dividend rate can be replaced by the foreign interest rate to calculate the value of an option on currencies

## payoffs of a vanilla call and put
def payoff_call(S, K):
    return max(S - K, 0)

def payoff_put(S, K):
    return max(K - S, 0)

## binomial model for european and american vanilla options
def european_option_tree(payoff, S, K, T, N):
    u = math.exp(sigma * math.sqrt(T/N))    # up factor
    d = math.exp(-sigma * math.sqrt(T/N))   # down factor
    a = math.exp(r-q)**(T/N)  # factor used to calculculate the up probility
    p = (a-d)/(u-d)           # probability of going up
    # we define a dict that we will use with keys (k ,i) such that k is the time step (0<= k <=N) and i the number of times by which the stock price went down, values in the dict are the value of the option
    Tree = {}
    # at the step N the option is execised or not, its value is the payoff of the option
    for i in range(N+1):
        Tree[(N, i)] = payoff(u**(N-i) * d**i *S, K)
    # backwards induction
    for k in range(N-1,-1,-1):
        for i in range(k+1):
            # optimization: if value at a certain node is already calculated, no need to do it again
            if not (k, i) in Tree:
                Tree[(k, i)] = (p*Tree[(k+1, i)] + (1-p)*Tree[(k+1, i+1)])*disc**(T/N)
    return Tree

def american_option_tree(payoff, S, K, T, N):
    u = math.exp(sigma * math.sqrt(T/N))    # up factor
    d = math.exp(-sigma * math.sqrt(T/N))   # down factor
    a = math.exp(r-q)**(T/N)
    p = (a-d)/(u-d)           # probability of going up
    tree = {}
    for i in range(N+1):
        tree[(N, i)] = payoff(u**(N-i) * d**i *S, K)

    for k in range(N-1,-1,-1):
        for i in range(k+1):
            if not (k, i) in tree:
                tree[(k, i)] = max(payoff(u**(k-i) * d**i * S, K), (p*tree[(k+1, i)] + (1-p)*tree[(k+1, i+1)])*disc**(T/N) )
    return tree

def european_option_value(payoff, S, K, T, N):
    return european_option_tree(payoff, S, K, T, N)[(0,0)]

def american_option_value(payoff, S, K, T, N):
    return american_option_tree(payoff, S, K, T, N)[(0,0)]

# display the tree: we generated a .html file that contains a JavaScript code displaying the tree
def drawTree(tree, payoff, S, K, T, N):
    u = math.exp(sigma * math.sqrt(T/N))
    d = math.exp(-sigma * math.sqrt(T/N))
    prices = [[u**(k-i) * d**i * S0 for i in range(k+1)] for k in range(N+1)]
    assert(N <= 10)  # Number of steps must be less than 10
    # dimensions used for drawing the rectangles containing option values and stock price
    w = 80
    h= 20
    D= 400
    f = open("Binomial_Tree.html", "w+")
    f.write('<!DOCTYPE html> \n<html> \n    <head></head>\n    <body>\n        <canvas id="tree" width="1500" height="1000"></canvas>')
    f.write('        <script>\n            var canvas = document.getElementById("tree");\n            var ctx = canvas.getContext("2d");\n             ctx.font = "12px Verdana";')
    for k in range(N+1):
        for i in range(k+1):
            f.write(f'            ctx.rect({w*k + 30*k}, {(D - 2*h*k) + 2*2*h*i}, {w}, {2*h});\n            ctx.stroke();')
            f.write(f'            ctx.moveTo({w*k + 30*k}, {(D - 2*h*k) + 2*2*h*i + h}); \n            ctx.lineTo({w*k + 30*k + w}, {D - 2*h*k + 2*2*h*i + h}); \n            ctx.stroke();')
            if k<N:
                f.write(f'            ctx.moveTo({w*k + 30*k + w}, {(D - 2*h*k) + 2*2*h*i + h}); \n            ctx.lineTo({w*(k+1) + 30*(k+1)}, {(D - 2*h*(k+1)) + 2*2*h*i + h }); \n            ctx.stroke();')
                f.write(f'            ctx.moveTo({w*k + 30*k + w}, {(D - 2*h*k) + 2*2*h*i + h}); \n            ctx.lineTo({w*(k+1) + 30*(k+1)}, {(D - 2*h*(k+1)) + 2*2*h*(i+1) + h }); \n            ctx.stroke();')
            # stock price
            f.write(f'            ctx.fillText("{format(prices[k][i],".4f")}", {w*k + 30*k + 5}, {(D - 2*h*k) + 2*2*h*i + .7*h});')
            # option value
            # if option value at time t is equal to its payoff at that time, then it must be exersised: to visualize that, display value in red
            if tree[(k, i)] == payoff(u**(k-i) * d**i * S, K) and tree[(k, i)] != 0:
                    f.write('\n            ctx.fillStyle="red";')
            f.write(f'            ctx.fillText("{format(tree[(k, i)],".6f")}", {w*k + 30*k + 5}, {(D - 2*h*k) + 2*2*h*i + 1.7*h});')
            f.write('\n            ctx.fillStyle="black";')

    f.write('        </script>\n    </body>\n</html>')
    f.close()

# quelques grecques à l'instant initial
def delta(option_tree, S0, T, N):
    u = math.exp(sigma * math.sqrt(T/N))
    d = math.exp(-sigma * math.sqrt(T/N))
    return (option_tree[(1, 0)] - option_tree[(1, 1)])/(S0*u - S0*d)

def gamma(option_tree, S0, T, N):
    u = math.exp(sigma * math.sqrt(T/N))
    d = math.exp(-sigma * math.sqrt(T/N))
    return ((option_tree[(2, 0)] - option_tree[(2, 1)])/(S0*u**2 - S0) - (option_tree[(2, 1)] - option_tree[(2, 2)])/(S0 - S0*d**2)) / (.5 * (S0*u**2 - S0*d**2))

## Black Scholes Merton Analytical model for european options
norm_cdf = lambda x : .5*(1 + math.erf(x/math.sqrt(2)))

def Black_Scholes_call(S, K, T):
    d1 = (math.log(S/K) + (r - q + sigma**2/2)*T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N1 = norm_cdf(d1)
    N2 = norm_cdf(d2)
    return S * N1 * math.exp(-q*T) - K * (disc**T) * N2

def Black_Scholes_put(S, K, T):
    d1 = (math.log(S/K) + (r - q + sigma**2/2)*T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N1 = norm_cdf(-d1)
    N2 = norm_cdf(-d2)
    return - S * N1 * math.exp(-q*T) + K * (disc**T) * N2

## Tests:

### You can modify the stock price, the strike price and time to maturity here
S0 = 50   # stock price
K = 40    # strike price
T = 1     # time to maturity

### convergeance of european binomial to Black Scholes
N = 1000   # number of tree steps
c = european_option_value(payoff_call, S0, K, T, N)
p = european_option_value(payoff_put, S0, K, T, N)
BS_call = Black_Scholes_call(S0, K, T)
BS_put = Black_Scholes_put(S0, K, T)

print("Number of steps used in Binomial Model: N = ", N)
print("European call Binomial: ", c )
print("European call Analytic (BS): ", BS_call)
print("European put Binomial:  ", p)
print("European put Analytic (BS): ", BS_put)
print("Call-put parity for binomial: ", "c-p=",c-p,"\n       S0 exp(-qT)- K exp(-rT)=", S0 * math.exp(-q*T) - K* disc**T)


print('American call binomial: ', american_option_value(payoff_call,S0,K,T,N))
print('American put binomial: ', american_option_value(payoff_put,S0,K,T,N))

### display binomial tree for american option
N = 10  #must be less than or equal to 10 to be able to display tree
american_put_tree = american_option_tree(payoff_put, S0, K, T, N)
drawTree(american_put_tree, payoff_put, S0, K, T, N)
#american_call_tree = american_option_tree(payoff_call, S0, K, T, N)
#drawTree(american_call_tree, payoff_call, S0, K, T, N)
webbrowser.open_new_tab('Binomial_Tree.html')


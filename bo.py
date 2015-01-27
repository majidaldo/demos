from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

""" This is code for simple GP regression. It assumes a zero mean GP Prior """

# This is the true unknown function we are trying to approximate
#f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 50         # number of test points.
sn = 0#.05    # noise variance.

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, N).reshape(-1,1)
# Sample some input points and noisy versions of the function evaluated at
# these points. 
#X = np.random.uniform(-5, 5, size=(n,1))
#yall = f(Xtest) + sn*np.random.randn(N)


from numpy import random
import scipy
from scipy import interpolate
def randomfunction(N=N):
    """makes a smooth squiggly function """
    #number of pts to go through  ~num of curves
    nc=np.random.choice(10)+5 #from 0 to 15 dont add too many pts
    #with these vales
    ys=np.random.randn(nc)
    #sprinkle it in the domain
    xis=np.random.choice(N, size=nc, replace=False) 
    xis.sort()
    xs=Xtest[xis].reshape(nc) #get rid of a dim
    spl= scipy.interpolate.InterpolatedUnivariateSpline( xs,ys
    )#,s=.1 )
    return spl(Xtest[:,0])

def init_randomfuction():
    global yall, ixmax #sinner!
    yall=randomfunction()
    ixmax=np.argmax(yall)

def init_compute():
    global computedis
    computedis=[]

def compute(ipt):#index of point
    global K, L, Lk, mu, K_, s2, s, y 
    if ipt in computedis: return    
    if not 0<=ipt<N: raise ValueError('index not in range')
    computedis.append(ipt)
    
    y= yall[computedis]
    X=Xtest[computedis]
    K = kernel(X, X)
    L = np.linalg.cholesky(K + sn*np.eye(len(computedis)))
    # compute the mean at our test points.
    Lk = np.linalg.solve(L, kernel(X, Xtest))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))

    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest)
    s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
    s = np.sqrt(s2)

#initial point.. which shouldn't be at the max
def init_initpt():
    global ip
    ip=np.random.choice(N)
    while(ip==ixmax):
        ip=np.random.choice(N)
    compute(ip) 

import scipy as sp
from scipy import stats
def PI(ix,ixp): #xp is 'encumbent' ..using indices
    eta=10
    return sp.stats.norm.cdf((mu[ix]-yall[ixp]-eta)/s[ix])
def maxiPI(): 
    PIs=(  PI(None
    , computedis[np.argmax(yall[computedis])] ))[0] #toss a dim
    
    PIs[computedis]=0 #if i know it then no improvement duh
    return np.argmax(PIs)

def ismax(ipt,tol=.02):
    """lets you know if a point is max"""
    #if not 0<=ipt<=N: raise ValueError('index not in range')
    #if ipt in computedis: return
    #if ipt==ixmax: return True
    tol=tol*(max(Xtest)-min(Xtest))
    if Xtest[ipt]<Xtest[ipt]<Xtest[ipt]+tol:
        return True
    else:
        compute(ipt) 
    return False

init_randomfuction()
init_compute()
init_initpt()

def play(player):
    n=0
    while True:
        guess=ismax(player.guess())
        n+=1
        if guess==True: return n
        else: continue


class player(object):
    my_guesses=[]

class human(player):
    
    def __init__(self):
        fig = pl.figure()
        self.cid = fig.canvas.mpl_connect('button_press_event'
        , lambda event: self.guess(event))
        pl.ylim((min(yall),max(yall))) #!
        pl.xlim((min(Xtest),max(Xtest)))
        pl.plot(Xtest[ip],[yall[ip]],'bo')
        pl.show(block=True)
        
   # @staticmethod
    #def onclick(event):
     #   self.guess(event)
#    @staticmethod
#    def verguess(guess):
#        if type(guess) is str: 
    def guess(self,event):
        while True:
            #pl.show();pl.draw()
            g=(event.xdata)
            if g.isdigit()==False:
                print 'input integer'
                continue
            g=int(g)
            if g== ixmax:
                print 'initial guess given'
                continue
            if g in self.my_guesses:
                print 'already guessed'
                continue
            if (0<=g<N)==False:
                print 'not in range'
                continue
            self.my_guesses.append(g)
            pl.plot([g],[yall[g]],'bo')
            break
        return g


#if __name__=='__main__': play(human())

## PLOTS:
#pl.figure(1)
#pl.clf()
#pl.plot(X, y, 'r+', ms=20)
#pl.plot(Xtest, f(Xtest), 'b-')
#pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
#pl.plot(Xtest, mu, 'r--', lw=2)
#pl.savefig('predictive.png', bbox_inches='tight')
#pl.title('Mean predictions plus 3 st.deviations')
#pl.axis([-5, 5, -3, 3])


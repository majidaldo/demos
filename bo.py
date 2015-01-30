from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
pl.ion()#ioff()#.ion()

""" bayesian optimizer game  """

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
    #number of pts to go through  ~num of knots
    nc=np.random.choice(12)+7 #from 7 to 19. dont add too many pts
    #with these vales
    ys=np.random.randn(nc)
    #sprinkle it in the domain
    xis=np.random.choice(N, size=nc, replace=False)
    xis.sort();
    #dont want edges flyhing off
    ys[0]=ys[1]; 
    ys[-1]=ys[-2]
    xis[0]=0
    xis[-1]=N-1
    xs=Xtest[xis].reshape(nc) #get rid of a dim
    spl= scipy.interpolate.InterpolatedUnivariateSpline( xs,ys
    )#,s=.1 )
    return spl(Xtest[:,0])

def init_randomfuction():
    global yall, ixmax #sinner!
    yall=randomfunction()
    ixmax=np.argmax(yall)

def init_u(etav=.01):#utility funciton
    global eta
    eta=etav

def init_compute():
    global computedis
    computedis=[]

def compute(ipt):#index of point
    global K, L, Lk, mu, K_, s, y 
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
    while(ismax(ip) == True):
        ip=np.random.choice(N)
    compute(ip)

import scipy as sp
from scipy import stats
def PI(ix,ixp): #xp is 'encumbent' ..using indices
    #global eta
    #eta=.05 #doesn't make much diff
    return sp.stats.norm.cdf( (mu[ix]-yall[ixp]-eta)/(s[ix]+1e-6)  )
def maxiPI():
    global ixp
    ixp= computedis[np.argmax(yall[computedis])]
    PIs=(  PI(None, ixp))[0] #toss a dim.
    PIs[computedis]=-1e-6 #if i know it then no improvement duh
    #PIl.append(PIs)
    return np.argmax(PIs)
#see
#A Tutorial on Bayesian Optimization of
#Expensive Cost Functions, with Application to
#Active User Modeling and Hierarchical
#Reinforcement Learning
#Eric Brochu, Vlad M. Cora and Nando de Freitas
#for info about these utility funcitons
def EI(ix,ixp): #xp is 'encumbent' ..using indices
    mu_y_eta=mu[ix]-yall[ixp]-eta
    Z=mu_y_eta/(s[ix]+1e-6)
    return mu_y_eta*sp.stats.norm.cdf(Z) + s[ix]*sp.stats.norm.pdf(Z)
def maxiEI():
    global ixp
    ixp= computedis[np.argmax(yall[computedis])]
    PIs=(  PI(None, ixp))[0] #toss a dim.
    PIs[computedis]=-1e-6 #if i know it then no improvement duh
    #PIl.append(PIs)
    return np.argmax(PIs)


def ismax(ipt,tol=.02):
    """lets you know if a point is max"""
    #if not 0<=ipt<=N: raise ValueError('index not in range')
    #if ipt in computedis: return
    #if ipt==ixmax: return True
    tol=tol*(Xtest[-1][0]-Xtest[0][0])
    if Xtest[ixmax]-tol<Xtest[ipt]<=Xtest[ixmax]+tol: #equals is important!..
                                    #...what if tol ~=0 ?
        return True
    else:
        compute(ipt) #this shouldnt be here
    return False

def init_all(kwargs):
    kwargs.setdefault('rf',{})
    kwargs.setdefault('compute',{})
    kwargs.setdefault('u',{})
    kwargs.setdefault('ip',{})
    init_randomfuction(**kwargs['rf'])
    init_compute(**kwargs['compute'])
    init_u(**kwargs['u'])
    init_initpt(**kwargs['ip'])

def play(player,initkw={}):
    #global PIl
    #PIl=[]
    init_all(initkw)
    n=0
    while True:
        guess=ismax(player.guess())
        n+=1;
        if n==N: return None #todo raise exception
        if guess==True: return n
        else: continue

#def compare


#you cant heuristcall program it

class player(object):
    def __init__(self):
        self.my_guesses=[]
        

class puter(player):
    
    def guess(self):
        gs=maxiEI()#maxiPI() #didn't see much differece
        self.my_guesses.append(gs)
        return self.my_guesses[-1]

    
class human(player):
    #todo plt thin vertical lines
    def __init__(self):
        super(human, self).__init__()
        self.fig = pl.gcf();
        pl.clf()
        self.pcid = self.fig.canvas.mpl_connect('button_press_event' 
                , lambda event: self.guessclick(event))
        
    def setupplay(self):

        pl.xlim((min(Xtest)-.5,max(Xtest)+.5))
        pl.title("Guess where the max is")
        pl.plot(Xtest[ip],[yall[ip]],'bo')
        mx=max(yall)
        mn=min(yall)
        m=np.random.uniform(.5,1) #players shouldn't know when ..
        #..they are close to the max
        mx=max(yall)
        mn=min(yall)
        pl.ylim((mn-m*(-mn+mx)
                ,mx+m*(-mn+mx) )) #+some margin
        for apt in Xtest: pl.plot([apt,apt],[mn-m*(-mn+mx),mx+m*(-mn+mx)]
            ,color='.2',lw=.2)
        #pl.show(block=False)
        #self.guess_clicked=False

        #rcid = self.fig.canvas.mpl_connect('button_release_event' 
        #        , lambda event: self.guessrelease(event))
        return

    def guess(self):
        if len(self.my_guesses)==0: #sigh hacky
            self.setupplay();
        #m=.5 #a margin
        #mx=max(yall[computedis])
        #mn=min(yall[computedis])
        #pl.ylim((mn-m*(-mn+mx)
        #        ,mx+m*(-mn+mx) )) #+some margin
        #self.cid = self.fig.canvas.mpl_connect('button_press_event' 
        #    , lambda event: self.guessclick(event))
        #pl.show(block=False)
        while ( pl.waitforbuttonpress(timeout=-1) ==False ): #false is mouse
            try:
                if ( self.guesschk(self.last_click)==True ): break
            except: pass
            else: continue
        #while (self.guess_clicked==False):# continue
            #pl.pause(1)
            #sleep(.1);# print 'not clicked'
            #self.fig.canvas.mpl_disconnect(cid)
        #while(self.guesschk(self.last_click)==False): continue
        igs=self.last_click
        self.my_guesses.append(igs)
        #self.fig.canvas.mpl_disconnect(self.cid)
        pl.plot([Xtest[igs]],[yall[igs]],'bo')
        return self.my_guesses[-1]

    def guessclick(self,event):
        try: self.last_click=np.abs(Xtest - event.xdata).argmin()
        except:
            self.last_click=None
            return None
        return self.last_click
    #def guessrelease(self,event):
    #    self.guess_clicked=True
    #    return
    
    def guesschk(self,ig):
        #while True:
            #pl.show();pl.draw()
        #g=Xtest[ig]
        if ig== ip:#todo except no coorrds (nonetype)
            print 'initial guess given'
            return False
        if ig in self.my_guesses:
            print 'already guessed'
            return False
        if (min(Xtest)<=Xtest[ig]<=max(Xtest))==False: #never comes here...
            print 'not in range' #..but i left these two lines
            return False
        return True
        #self.my_guesses.append(g)
            #pl.plot([ig],[yall[ig]],'bo'); pl.show()
            #break
        #return g
#
#
#qt=123213
#while(qt is not None):
#    p=puter()
#    qt=play(p)


#hp=human()
#ptr=puter()
#hp.guess()
#print play(ptr)
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


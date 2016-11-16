import numpy as np
#import pandas
import matplotlib.pyplot as plt


def wiener():
    dt = 0.1
    T = 1.
    w = np.array([0.])

    for i in range(1,int(T/dt)):
        w = np.append(w,[w[-1] +dt**0.5*np.random.randn()])
    return w

#nntaleb
t = np.linspace(0,1.,10)
"""
u_mean = np.zeros(100)
for i in range(1000):
    u_mean += np.exp(t+0.5*wiener())/1000.
    
plt.plot(t,u_mean)
plt.plot(t,np.exp(9.*t/8.))
plt.show()
"""

x = np.array([307.65])
w = wiener()
k = 0.75
s = 0.3
for i in range(1,10):
    x = np.append(x,[x[-1]+k*x[-1]*0.1+s*x[-1]*(w[i]-w[i-1])])

plt.plot(t,x)
plt.plot(t,x[0]*np.exp((k-0.5*s**2)*t+s*w))
plt.show()

"""ra = np.random.random((5000,2))
data = pandas.DataFrame(data=ra,columns=['a','b'])
#pandas.DataFrame.hist(data)
data.plot(kind='hexbin',x='a',y='b')
plt.show()
"""

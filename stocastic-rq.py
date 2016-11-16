import numpy as np
#import pandas
import matplotlib.pyplot as plt


def wiener():
    dt = 0.01
    T = 1.
    w = np.array([0.])

    for i in range(1,int(T/dt)):
        w = np.append(w,[w[-1] +dt**0.5*np.random.randn()])
    return w

t = np.linspace(0,1.,100)
u_mean = np.zeros(100)
for i in range(1000):
    u_mean += np.exp(t+0.5*wiener())/1000.
    
plt.plot(t,u_mean)
plt.plot(t,np.exp(9.*t/8.))
plt.show()

"""ra = np.random.random((5000,2))
data = pandas.DataFrame(data=ra,columns=['a','b'])
#pandas.DataFrame.hist(data)
data.plot(kind='hexbin',x='a',y='b')
plt.show()
"""

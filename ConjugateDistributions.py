#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import beta


# In[2]:


def Generatedataset():
    k = 150
    ar = np.random.permutation(160)
    dataset = []
    for i in ar:
        if i < k:
            dataset.append(1)
        else:
            dataset.append(0)

    return k,dataset

def gamma(a):
    return scipy.special.gamma(a)

def parta(i,dataset,a,b):
    size = 1000
    x = np.random.random([size,1])
    x = x**4
    x = np.sort(x,axis=None)
    y = ((x**(a-1))*((1-x)**(b-1))*gamma(a+b))/(gamma(a)*gamma(b))
    #Beta(k+a,160+b-k)
    plot(x,y,'PartA_'+str(i))
    if i<160:
        parta(i+1,dataset,a+dataset[i],b+1-dataset[i])


def plot(x,y,c):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(x.min(), x.max(), 1000)

    spl = make_interp_spline(x,y, k=3)  # type: BSpline
    power_smooth = spl(xnew)

    plt.ylim(top=20)
    plt.xlabel('Probability of occurence of heads')
    plt.ylabel('Pdf')
    plt.plot(xnew, power_smooth)
#     plt.savefig(c+'.png')
    plt.show()


def partb(k,dataset,a,b):
    print("PART B ---  LISA's SOLUTION")
    size = 1000
    x = np.random.random([size,1])
    x = x**4
    x = np.sort(x,axis=None)
    y = ((x**(k+1))*((1-x)**(162-k))*gamma(165))/(gamma(k+2)*gamma(163-k))
    plot(x,y,'u = 0.5')

if __name__== "__main__":
    k,dataset = Generatedataset()
    a = 2
    b = 3
    print("k = ",k)
    print(dataset)

    parta(0,dataset,a,b)
    partb(k,dataset,a,b)


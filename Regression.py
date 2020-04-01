#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ASSIGNMENT -- 2

# POLYNOMIAL REGRESSION
# Captal X -- data taken from dataset
# small x  -- x matrix with all possible combination for nth order 
# learn_rate for 6 degree is 3x10^(-9)
# learn_rate for 5 degree is 3x10^(-9)
# learn_rate for 4 degree is 3x10^(-8)
# learn_rate for 3 degree is 3x10^(-7) 1000
# learn_rate for 2 degree is 3x10^(-7) 1000
# learn_rate for 1 degree is 3x10^(-6) 


# In[2]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm,trange,tqdm_notebook


# In[3]:


data = pd.read_csv(r"R:\3-1\CS F320\3D_spatial_network.csv")


# In[4]:


X = np.array(data.iloc[:,0:2].values)

y = np.array(data.iloc[:,2:3].values)
learn_rate  = 0.000000003


# In[5]:



def error_r(w,x,y,Lambda,s):
    
    p = np.matmul(x,w)

    error = 0.5*np.matmul(np.transpose(p-y),(p-y))

    
    if(s == 'l1'):
        error +=Lambda*np.abs(w).sum()
        return error
    
    if(s =='l2'):
        error +=Lambda*np.square(w).sum()
        return error
    


# In[6]:


# NORMALISE

def normalise():
    
    global X,y
    
    x1_bar = X[:,0].mean()
    x1_std = X[:,0].std()
    

    x2_bar = X[:,1].mean()
    x2_std = X[:,1].std()
    

    
    X[:,0] = (X[:,0] - x1_bar) / (x1_std)
    X[:,1] = (X[:,1] - x2_bar)/(x2_std)

    
normalise();


# In[7]:


# Dividing data to training and testing

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

nTrain = yTrain.size
nTest  = yTest.size


# In[8]:



def update_r(w,x,y,Lambda,s):
    
    p = np.matmul(x,w) 
    
    error_w = np.matmul(np.transpose(x),np.subtract(p,y))

    if(s == 'l1'):
        w_df = np.copy(w)
        w_df[w_df<0]=-1
        w_df[w_df>0]=1
        

        error_w += Lambda*w_df
        
    
    
    if(s=='l2'):
        error_w +=  2*Lambda*w
        

   
    w -= learn_rate*error_w
    
    
    return w
    


# In[9]:


def polyregress(x,yTrain,n,Lambda,s):
    
    count =0;
    
    N = yTrain.size
    x = np.zeros((N,1))
    x = np.delete(x,0,1)    

    for k in tqdm_notebook(range(n+1)):
        for i in range(n+1-k):
            for j in range(n+1-k):
                if(i+j==n-k):

                    p = np.multiply(np.power(xTrain[:,0:1],i),np.power(xTrain[:,1:2],j))
                    
                    x = np.insert(x,[count],p,axis=1)
        
                    count = count+1



    x = np.fliplr(x)
    


    
    
    w = np.random.randint(1,100,size=(count,1))*0.01

    for i in tqdm_notebook(range(10000)):
        
        w = update_r(w,x,yTrain,Lambda,s)
        
        
    error = error_r(w,x,yTrain,Lambda,s)
        
    print('\nModel:\n\n ',w,'\n\nHalf sum of squares: ',error,'\nMean Sqaured Error: ',error/N)
    
    return w


# In[10]:


def test(w,x,yTest,n,Lambda,s): # n is the order 
        
        
        
        count =0;
    
        N = yTest.size
        x = np.zeros((N,1))
        x = np.delete(x,0,1)    

        for k in range(n+1):
            for i in range(n+1-k):
                for j in range(n+1-k):
                    if(i+j==n-k):

                        p = np.multiply(np.power(xTest[:,0:1],i),np.power(xTest[:,1:2],j))

                        x = np.insert(x,[count],p,axis=1)
                        count = count+1



        x = np.fliplr(x)
        
        
        
        R2 = R_squared(w,x,yTest)
        
        
        RMSE = np.sqrt(2*error_r(w,x,yTest,Lambda,s)/N)

        return RMSE,R2


# In[11]:


def R_squared(w,x,y):
    
    y_bar = np.copy(np.mean(y))
    
    
    ss_total = np.square(y-y_bar).sum()
   
    p = np.matmul(x,w)
    
    ss_reg = np.square(p-y).sum()
    
    
    
        
    R_2 = 1-(ss_reg/ss_total)
    
    return R_2


# In[12]:



'''
 Part-A and B
 
 Polyregress() function recieves the xdata,ydata,degree of the polynomial,Lambda,Type of regularisation respectively
 as input
 
'''
n = 1
Lambda = 0
print("Degree:",n,'\n')

w_final = polyregress(xTrain,yTrain,n,Lambda,'l2')

rmse,r2 = test(w_final,xTest,yTest,n,Lambda,'l2')

print('\n\nOn the Test Data\nRMSE: ',rmse,'\nR2: ',r2)


# In[13]:



    '''
    PART C
    
      l1 and l2 regularisation of Nth degree polynomial
    
    '''
def binarySearchLambda(xTrain,yTrain,xTest,yTest,n,s):
    
    
    #Binary search for Lambda
    l = 0
    r = 1000
    count=0
    RMSE = np.array([])
    Lambda = np.array([])

    while(l<=r):

        mid = (l + r)/2

        print(mid)


        Lambda = np.append(Lambda,mid)

        w_final = polyregress(xTrain,yTrain,n,mid,s)

        rmse_mid,r2_mid = test(w_final,xTest,yTest,n,mid,s)

        RMSE = np.append(RMSE,rmse_mid)

        count= count +1


    #RMSE FOR RIGHT HALF
    
        c = 0.01*mid

        w_final = polyregress(xTrain,yTrain,n,mid+c,s)

        rmse_r,r2_r =  test(w_final,xTest,yTest,n,mid+c,s)


    # RMSE FOR LEFT HALF

        w_final = polyregress(xTrain,yTrain,n,mid-c,s)

        rmse_l,r2_l =  test(w_final,xTest,yTest,n,mid-c,s)    


        if(rmse_r < rmse_mid):
            l = mid



        if(rmse_l < rmse_mid):

            r = mid
            
        if(rmse_mid < rmse_l and rmse_mid < rmse_r):
            break


        if(count==10):
            break
        
    plt.plot(Lambda,RMSE)
    plt.ylabel('RMSE')
    plt.xlabel('Lambda')
    plt.title(s+'Regularisation')
    plt.show() 

# binarySearchLambda(xTrain,yTrain,xTest,yTest,6,'l2')


    


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy
import pandas


import matplotlib.pyplot as plt



# In[38]:


dataframe = pandas.read_csv("burnforest.csv")



# Encode Data
dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)


# In[39]:


print("Head:", dataframe)


# In[40]:


print("Statistical Description:", dataframe.describe())


# In[41]:


dataset = dataframe.values

print(dataset)
X = dataset[:,0:8]
Y = dataset[:,12]
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.2,random_state =2)
dataframe


# In[62]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
Et = ExtraTreesRegressor()
from sklearn.tree import DecisionTreeRegressor
Dt=DecisionTreeRegressor()

Et.fit(X,Y)



# In[63]:



# In[64]:


result = Et.score(Xtest, Ytest)


# In[65]:


result 


# In[66]:


a=[1,2,3,0,5,6,7]
print(min(a))


# In[67]:


import joblib
filename = 'm2.pkl'
joblib.dump(Et, filename)


# In[68]:


m2 = joblib.load('m2.pkl')


# In[69]:


result = m2.score(Xtest, Ytest)
print(result)


# In[53]:







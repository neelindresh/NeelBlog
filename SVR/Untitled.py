
# coding: utf-8

# In[67]:


import pandas as pd


# In[68]:


df=pd.read_csv('./age_mod.csv')


# In[69]:


df.head()


# In[70]:


df=df.drop(['Sex'],axis=1)


# In[71]:


from sklearn.svm import SVR


# In[72]:


regressor=SVR(kernel='linear',degree=1)


# In[73]:


import matplotlib.pyplot as plt


# In[74]:


plt.scatter(df['Shucked weight'],df['Age'])


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y)


# In[77]:


regressor.fit(xtrain,ytrain)


# In[78]:


pred=regressor.predict(xtest)


# In[80]:


print(regressor.score(xtest,ytest))


# In[83]:


from sklearn.metrics import r2_score


# In[85]:


print(r2_score(ytest,pred))


# In[98]:


regressor=SVR(kernel='rbf',epsilon=1.0)
regressor.fit(xtrain,ytrain)
pred=regressor.predict(xtest)
print(regressor.score(xtest,ytest))
print(r2_score(ytest,pred))


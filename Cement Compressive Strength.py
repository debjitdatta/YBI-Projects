#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


cement = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv')


# In[3]:


cement.head()


# In[4]:


cement.info()


# In[5]:


cement.describe()


# In[6]:


cement.columns


# In[7]:


y=cement['Concrete Compressive Strength(MPa, megapascals) ']
X=cement[['Cement (kg in a m^3 mixture)',
       'Blast Furnace Slag (kg in a m^3 mixture)',
       'Fly Ash (kg in a m^3 mixture)', 'Water (kg in a m^3 mixture)',
       'Superplasticizer (kg in a m^3 mixture)',
       'Coarse Aggregate (kg in a m^3 mixture)',
       'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)']]


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4548)


# In[9]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[10]:


model.fit(X_train,y_train)


# In[11]:


y_pred=model.predict(X_test)


# In[12]:


model.predict(X_test)


# In[13]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[14]:


mean_absolute_error(y_test,y_pred)


# In[15]:


mean_absolute_percentage_error(y_test,y_pred)


# In[16]:


mean_squared_error(y_test,y_pred)


# In[ ]:





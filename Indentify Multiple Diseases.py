#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd


# In[22]:


disease = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MultipleDiseasePrediction.csv')


# In[23]:


disease.head()


# In[24]:


disease.info()


# In[25]:


disease.describe()


# In[26]:


disease.columns


# In[27]:


y = disease['prognosis']
X = disease[[ 'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
       'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
       'ulcers_on_tongue','blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
       'small_dents_in_nails', 'inflammatory_nails', 'blister',
       'red_sore_around_nose', 'yellow_crust_ooze' ]]


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size = 0.7, random_state = 2529)


# In[29]:


#step 4 :Train Test split
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[35]:


# Step 5 : select model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[36]:


model.fit(X_train,y_train)


# In[37]:


y_pred=model.predict(X_test)


# In[38]:


#Step 8 : model accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[39]:


accuracy_score(y_test,y_pred)


# In[40]:


confusion_matrix(y_test,y_pred)


# In[41]:


print(classification_report(y_test,y_pred))


# In[ ]:





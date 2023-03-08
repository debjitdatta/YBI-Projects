#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


employee = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/EmployeeAttrition.csv')


# In[3]:


employee.head()


# In[4]:


employee.info()


# In[5]:


employee.describe()


# In[6]:


employee.columns


# In[25]:


y=employee['Attrition']
X=employee[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']]


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4548)


# In[27]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[28]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[29]:


model.fit(X_train,y_train)


# In[30]:


y_pred=model.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[32]:


accuracy_score(y_test,y_pred)


# In[33]:


confusion_matrix(y_test,y_pred)


# In[34]:


print(classification_report(y_test,y_pred))


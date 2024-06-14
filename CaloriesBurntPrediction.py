#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[2]:


get_ipython().system('pip install xgboost')


# In[4]:


calories = pd.read_csv('E:/Zip files/archive/calories.csv')


# In[5]:


calories.head()


# In[6]:


exercise_data = pd.read_csv('E:/Zip files/archive/exercise.csv')


# In[7]:


exercise_data.head()


# In[8]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)


# In[9]:


calories_data.head()


# In[10]:


calories_data.shape


# In[11]:


calories_data.info()


# In[12]:


calories_data.isnull().sum()


# In[13]:


calories_data.describe()


# In[14]:


sns.set()


# In[17]:


sns.distplot(calories_data['Age'])


# In[18]:


sns.distplot(calories_data['Height'])


# In[19]:


sns.distplot(calories_data['Weight'])


# In[23]:


correlation = calories_data.corr(numeric_only=True)


# In[24]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[25]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[26]:


calories_data.head()


# In[27]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']


# In[28]:


print(X)


# In[29]:


print(Y)


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# In[32]:


model = XGBRegressor()


# In[33]:


model.fit(X_train, Y_train)


# In[34]:


test_data_prediction = model.predict(X_test)


# In[35]:


print(test_data_prediction)


# In[36]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[37]:


print("Mean Absolute Error = ", mae)


# In[ ]:





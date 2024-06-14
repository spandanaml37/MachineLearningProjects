#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


pip show pandas


# In[4]:


import nltk
nltk.download('stopwords')


# In[5]:


print(stopwords.words('english'))


# In[6]:


news_dataset = pd.read_csv('E:/Zip files/fake-news/train.csv')


# In[7]:


news_dataset.shape


# In[8]:


news_dataset.head()


# In[9]:


news_dataset.isnull().sum()


# In[10]:


news_dataset = news_dataset.fillna('')


# In[11]:


news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']


# In[12]:


print(news_dataset['content'])


# In[13]:


X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']


# In[14]:


print(X)
print(Y)


# In[15]:


port_stem = PorterStemmer()


# In[16]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[17]:


news_dataset['content'] = news_dataset['content'].apply(stemming)


# In[18]:


print(news_dataset['content'])


# In[19]:


X = news_dataset['content'].values
Y = news_dataset['label'].values


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


Y.shape


# In[23]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[24]:


print(X)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[26]:


model = LogisticRegression()


# In[27]:


model.fit(X_train, Y_train)


# In[28]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[29]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[30]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[31]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[32]:


X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[33]:


print(Y_test[3])


# In[34]:


X_new = X_test[6]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[35]:


X_new = X_test[8]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[ ]:





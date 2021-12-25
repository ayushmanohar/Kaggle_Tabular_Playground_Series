#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


df


# In[4]:


df.drop(['Id'], axis='columns',inplace=True)


# In[5]:


df


# In[6]:


target = df.Cover_Type
df.drop(['Cover_Type'], axis='columns')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.3)


# In[9]:


len(x_train)


# In[10]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[11]:


model.fit(x_train, y_train)


# In[12]:


model.score(x_test, y_test)


# In[ ]:





# In[ ]:


import pickle


# In[ ]:


with open('model_pickle', 'wb') as f:
    pickle.dump(model,f)


# In[ ]:


with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)


# In[ ]:


test_data = pd.read_csv('test.csv')


# In[ ]:


test_data


# In[ ]:


y_pred = mp.predict(test_data)


# In[ ]:


y_pred


# In[ ]:


pred = pd.DataFrame(y_pred)


# In[ ]:


sub_df = pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'], pred], axis=1)
datasets.columns=['Id', 'Cover_Type']
datasets.to_csv('sample_submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





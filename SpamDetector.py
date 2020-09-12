#!/usr/bin/env python
# coding: utf-8

# In[67]:



import pandas as pd
import re
import nltk
nltk.download('stopwords')

df = pd.read_csv('C:/Users/rahul03/SpyderProjects/Natural_Language_Processing/spam_data.txt', sep='\t',
                           names=["label", "message"])

df.head()


# In[68]:


df['label'].value_counts()


# In[69]:


ham = df[df['label']=='ham']
ham.head()


# In[70]:


spam = df[df['label']=='spam']
spam.head()


# In[71]:


ham.shape, spam.shape


# In[72]:


ham = ham.sample(spam.shape[0])


# In[77]:


ham.shape, spam.shape


# In[78]:


data = ham.append(spam, ignore_index = True)


# In[79]:


data['label'].value_counts()


# In[80]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

    


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size = 0.30, shuffle = True, random_state = 0)


# In[89]:


y_train


# In[83]:


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_train.shape


# In[90]:


X_train


# In[91]:


clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf' , RandomForestClassifier(n_estimators=100))])
clf.fit(X_train, y_train)


# In[92]:


y_pred = clf.predict(X_test)


# In[93]:


confusion_matrix(y_test, y_pred)


# In[94]:


print(classification_report(y_test, y_pred))


# In[95]:


y_train.value_counts()


# In[96]:


accuracy_score(y_test, y_pred)


# In[100]:


clf.predict(['You have won free lottery'])


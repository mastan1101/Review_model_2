#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[20]:


yelp = pd.read_csv('./Downloads/yelp.csv')


# In[21]:


yelp.shape


# In[22]:


yelp.head()


# In[23]:


yelp['text length'] = yelp['text'].apply(len)
yelp.head()


# In[24]:


g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)


# In[25]:


sns.boxplot(x='stars', y='text length', data=yelp)


# In[26]:


stars = yelp.groupby('stars').mean()
stars.corr()


# In[27]:


sns.heatmap(data=stars.corr(), annot=True)


# In[28]:


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
yelp_class.shape


# In[49]:


X = yelp_class['text']
y = yelp_class['stars']


# In[50]:


import string
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[51]:


from sklearn.feature_extraction.text import CountVectorizer


# In[52]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(X)


# In[53]:


sample_text = "Hey there! This is a sample review, which happens to contain punctuations."
print(text_process(sample_text))
Output: ['Hey', 'sample', 'review', 'happens', 'contain', 'punctuations']


# In[54]:


len(bow_transformer.vocabulary_)


# In[55]:


review_25 = X[24]
review_25


# In[56]:


bow_25 = bow_transformer.transform([review_25])
bow_25


# In[57]:


print(bow_transformer.get_feature_names()[11443])
print(bow_transformer.get_feature_names()[22077])


# In[ ]:


X = bow_transformer.transform(X)


# In[61]:


print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz )
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))


# In[88]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[89]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[90]:


preds = nb.predict(X_test)


# In[91]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


# In[92]:


positive_review = yelp_class['text'][59]
positive_review


# In[93]:


positive_review_transformed = bow_transformer.transform([positive_review])
nb.predict(positive_review_transformed)[0]


# In[94]:


negative_review = yelp_class['text'][281]
negative_review


# In[85]:


negative_review_transformed = bow_transformer.transform([negative_review])
nb.predict(negative_review_transformed)[0]


# In[70]:


another_negative_review = yelp_class['text'][140]
another_negative_review


# In[71]:


another_negative_review_transformed = bow_transformer.transform([another_negative_review])
nb.predict(another_negative_review_transformed)[0]


# In[72]:


string="the restaurent is awesome and pretti good to eat meal and non veg is fabulus"
nb.predict(another_negative_review_transformed)[0]


# In[82]:


review ="the food is but the service is poor"
type(review)
negative_review_transformed1 = bow_transformer.transform([review])
# nb.predict(negative_review_transformed1)[0]


# In[ ]:





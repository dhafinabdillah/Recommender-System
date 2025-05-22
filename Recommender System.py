#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data sampling and management
import pandas as pd
pd.set_option('display.max_columns', None)

movie_df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/title.basics.tsv", sep='\t') 
rating_df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/title.ratings.tsv", sep='\t')


# In[2]:


movie_df.head()


# In[3]:


rating_df.head()


# In[4]:


# Data cleaning
movie_df.isnull().sum()


# In[5]:


rating_df.isnull().sum()


# In[6]:


movie_df.loc[(movie_df['primaryTitle'].isnull()) | (movie_df['originalTitle'].isnull())]


# In[7]:


movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull())]


# In[8]:


movie_df.loc[movie_df['genres'].isnull()]


# In[9]:


movie_df = movie_df.loc[(movie_df['genres'].notnull())]


# In[10]:


movie_df.isnull().sum()


# In[11]:


import numpy as np
movie_df['startYear'] = movie_df['startYear'].replace('\\N', np.nan)
movie_df['startYear'] = movie_df['startYear'].astype('float64')
print(movie_df['startYear'].unique()[:5])

movie_df['endYear'] = movie_df['endYear'].replace('\\N', np.nan)
movie_df['endYear'] = movie_df['endYear'].astype('float64')
print(movie_df['endYear'].unique()[:5])

movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].replace('\\N', np.nan)
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].astype('float64')
print(movie_df['runtimeMinutes'].unique()[:5])


# In[12]:


def transform_to_list(x):
    if ',' in x: 
    #ubah menjadi list apabila ada data pada kolom genre
        return x.split(',')
    else: 
    #jika tidak ada data, ubah menjadi list kosong
        return []

movie_df['genres'] = movie_df['genres'].apply(lambda x: transform_to_list(x))


# In[13]:


movie_rating_df = pd.merge(movie_df, rating_df, on='tconst', how='inner')


# In[14]:


movie_rating_df.dropna(subset=['startYear', 'runtimeMinutes'])


# In[15]:


# EDA
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=movie_rating_df, x='startYear', hue='titleType', bins=30, multiple='stack')
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=movie_rating_df, x='runtimeMinutes', y='averageRating', size='numVotes', hue='titleType', alpha=0.7, sizes=(20, 200))
plt.show()


# In[17]:


# Pre processing
def imdb_weighted_rating(df, var=0.8):
    v = df['numVotes']
    R = df['averageRating']
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(var)
    df['score'] = (v/(m+v))*R + (m/(m+v))*C
    return df['score']


# In[18]:


# Learning and result
imdb_weighted_rating(movie_rating_df)

m = movie_rating_df['numVotes'].quantile(0.8)

def simple_recommender(df, top=100):
    df = df.loc[df['numVotes'] >= m]
    df = df.sort_values(by='score', ascending=False)
    
    df = df[:top]
    return df
       
from itables import show
import itables.options as opt

opt.lengthMenu = [5, 10, 25, 50, 100]
opt.maxBytes = 0

from itables import show

top_movies = simple_recommender(movie_rating_df)

show(top_movies)


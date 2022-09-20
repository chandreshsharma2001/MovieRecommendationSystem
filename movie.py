# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:03:41 2022

@author: Chandresh Sharma
"""
import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

# print(movies.info())
# print(credit.info())

movies = movies.merge(credit, on='title')

# For based on popularity
popularity = movies[['id', 'original_title', 'popularity']]
print(popularity.info())
popularity = popularity.sort_values(['popularity'], ascending=False)[0:5]
popularity.drop(['popularity'], axis=1, inplace=True)
#####################
feature = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
movies = movies[feature]

# print(movies.isnull().sum())
# Only 3 columns in overview are missing so just drop them
movies = movies.dropna()


# for dublicates
# print(movies.duplicated().sum())

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert2)


def fetch(obj):
    L = []

    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_movies = movies[['movie_id', 'title', 'tag']]

new_movies['tag'] = new_movies['tag'].apply(lambda x: " ".join(x))
new_movies['tag'] = new_movies['tag'].apply(lambda x: x.lower())

# Stemming

import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def stem(X):
    L = []
    for i in X.split():
        L.append(ps.stem(i))

    return " ".join(L)


new_movies['tag'] = new_movies['tag'].apply(stem)

# Declaring TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(new_movies['tag']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

cosine = cosine_similarity(vectors)


# print(cosine.shape)

def recommend(X):
    movie_index = new_movies[new_movies['title'] == X].index[0]
    dist = cosine[movie_index]
    movies_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_movies.iloc[i[0]].title)


recommend('Batman Begins')

import pickle

pickle.dump(new_movies, open('movies.pkl', 'wb'))
pickle.dump(cosine, open('similarity.pkl', 'wb'))
pickle.dump(popularity, open('popularity.pkl', 'wb'))

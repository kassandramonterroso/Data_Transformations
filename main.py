import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

reviews = pd.read_csv('reviews.csv')
 
print(reviews.columns)
print(reviews.info())

print(reviews['recommended'].value_counts())
 
binary_dict = {True:1,False:0}
 
reviews['recommended'] = reviews['recommended'].map(binary_dict)

print(reviews['recommended'].value_counts())


print(reviews['rating'].value_counts())
 
rating_dict = {"Loved it":5, "Liked it":4, "Was okay":3, "Not great":2, "Hated it":1}
 
reviews['rating'] = reviews['rating'].map(rating_dict)

print(reviews['rating'].value_counts())


print(reviews['department_name'].value_counts())
 
ohe = pd.get_dummies(reviews['department_name'])

reviews = reviews.join(ohe)
print(reviews.columns)


reviews['review_date'] = pd.to_datetime(reviews['review_date'])

print(reviews['review_date'].dtype)


reviews = reviews[['clothing_id', 'age', 'recommended', 'rating', 'Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend']].copy()

reviews = reviews.set_index('clothing_id')

scaler = StandardScaler()
scaler.fit_transform(reviews)

print(reviews)






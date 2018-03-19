import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

yelp = pd.read_csv('yelp.csv')

# some basic information about the data.

# print(yelp.shape )
# print(yelp.describe())

# adding length column to determine the length of text

yelp['text length'] = yelp['text'].apply(len)
print(yelp.head())

# Relation between Text length and stars column's using Seaborn library:
#g = sns.FacetGrid(data=yelp, col='stars')
#print(g.map(plt.hist, 'text length', bins=50))

# finding Correlation using Pandas

stars = yelp.groupby('stars').mean()
print(stars.corr())

# To visualise these correlations, we can use Seaborn’s heatmap:
#print(sns.heatmap(data=stars.corr(), annot=True))

# New dataframe with reviews either 1 0r 5 star ratings from Yelp.

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]  # print(yelp_class.shape)

x = yelp_class['text']  # text column

y = yelp_class['stars']  # star column

# Text pre-processing

import string


def pre_process(text):
#         Takes in a string of text, then performs the following:
#         1. Remove all punctuation
#         2. Remove all stopwords
#         3. Return the cleaned text as a list of words
        

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# sample_text = "Hey there! This is a sample review, which happens to contain punctuations."
# print(pre_process(sample_text))


# vectorization
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=pre_process).fit(x)
review_25 = x[24]
print(review_25)

bow_25 = bow_transformer.transform([review_25])
print(bow_25)

x = bow_transformer.transform(x)
print('Shape of Sparse Matrix: ', x.shape)
print('Amount of Non-Zero occurrences: ', x.nnz)

# Percentage of non-zero values
density = (100.0 * x.nnz / (x.shape[0] * x.shape[1]))
print('Density: {}'.format((density)))

# training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, y_train)

# testing
preds = nb.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

# Predicting a singular positive review example

positive_review = yelp_class['text'][59]
positive_review_transformed = bow_transformer.transform([positive_review])
print(nb.predict(positive_review_transformed)[0])

# -*- coding: utf-8 -*-
"""j002-j014 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19f6aOiGuxViButb8SAl2F23Jbrw1bAuN

# Cyberbullying Detection using NLP
---

![](https://images.unsplash.com/photo-1585007600263-71228e40c8d1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1770&q=80)

# Imports
---
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import time

import joblib

"""# Loading the Dataset
---
"""

df = pd.read_csv('/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv', nrows = 20000)
df

df.head()

df.tail()

df.shape

"""# EDA
---

# 1. Handling Null Values
"""

df.isna().any()

df.isna().sum()

"""# 2. Handling Duplicate Values"""

df.nunique()

df['tweet_text'].nunique()

"""# 3. Class Distributions"""

df['cyberbullying_type'].value_counts()

# Create a bar plot of the class distribution
class_counts = df['cyberbullying_type'].value_counts()
class_counts.plot(kind='bar')
plt.title('Class Distribution of Cyberbullying Types')
plt.xlabel('Labels')
plt.ylabel('Number of Tweets')
plt.show()

"""# 4. Word Count"""

from collections import Counter
import re

import nltk
from nltk.corpus import stopwords

# Concatenate all tweet texts into a single string
all_text = ' '.join(df['tweet_text'].values)
# Remove URLs, mentions, and hashtags from the text
all_text = re.sub(r'http\S+', '', all_text)
all_text = re.sub(r'@\S+', '', all_text)
all_text = re.sub(r'#\S+', '', all_text)
# Split the text into individual words
words = all_text.split()

# Remove stop words
stop_words = set(stopwords.words('english'))
words = [word for word in words if not word in stop_words]

# Count the frequency of each word
word_counts = Counter(words)
top_words = word_counts.most_common(100)
top_words

# Create a bar chart of the most common words
top_words = word_counts.most_common(10) # Change the number to show more/less words
x_values = [word[0] for word in top_words]
y_values = [word[1] for word in top_words]
plt.bar(x_values, y_values)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Most Commonly Used Words')
plt.show()

"""# Visualizations
---

# 1. Wordclouds
"""

from wordcloud import WordCloud

text = ' '.join([word for word in df['tweet_text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=1000, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words used in Tweets', fontsize=19)
plt.show()

"""# 2. Sentiment Analysis Plot"""

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# perform sentiment analysis on each text in DataFrame
sentiment_scores = []
for text in df['tweet_text']:
    analysis = TextBlob(text)
    sentiment_scores.append((analysis.sentiment.polarity, analysis.sentiment.subjectivity))

# create DataFrame with sentiment scores
sentiment_df = pd.DataFrame(sentiment_scores, columns=['polarity', 'subjectivity'])

# plot distribution of sentiment scores
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sentiment_df['polarity'].plot(kind='hist', ax=axes[0], title='Polarity')
sentiment_df['subjectivity'].plot(kind='hist', ax=axes[1], title='Subjectivity')
plt.show()

"""# 3. Named Entity Recognition (NER) Plot"""

import spacy
from spacy import displacy

#sample text
text = df['tweet_text'].iloc[4]

#load pre-trained NER model
nlp = spacy.load('en_core_web_sm')

#perform named entity recognition
doc = nlp(text)

#visualize named entities
displacy.render(doc, style='ent', jupyter=True)

"""# 4. Part-of-Speech (POS) Tagging Plot"""

import spacy
from spacy import displacy

#sample text
text = df['tweet_text'].iloc[1]

#load pre-trained POS tagging model
nlp = spacy.load('en_core_web_sm')

#perform POS tagging
doc = nlp(text)

#visualize POS tagging
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})

"""# 5. Topic Modeling Visualization"""

!pip install pyLDAvis

import gensim
import pyLDAvis
from pyLDAvis import *
import pyLDAvis.gensim as gensimvis

# Preprocessing
tokens = [[word for word in sentence.split()] for sentence in df['tweet_text']]
dictionary = gensim.corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]

# Topic Modeling
num_topics = 10
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

# Visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)
vis

"""# Natural Language Processing
---

# 1. Data Cleaning
"""

# Clean the data
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    # Join the words back into a string
    text = ' '.join(words)
    return text

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# tqdm.pandas()
# 
# df['cleaned_text'] = df['tweet_text'].progress_apply(clean_text)

"""# 2. Feature Extraction"""

# Create the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_text']).toarray()
y = df['cyberbullying_type']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Classification Model
---

# 1. Logistic Regression Model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# train a Logistic Regression Model
logistic_regression_model = LogisticRegression(max_iter = 1000)

logistic_regression_model.fit(X_train, y_train)

# Save the model to an HDF5 file
model_filename = 'logistic_regression_model.h5'
joblib.dump(logistic_regression_model, model_filename)
print(f"Logistic Regression Model saved to {model_filename}")

"""# 2. Predictions"""

# evaluate the classifier on the test set
y_pred = logistic_regression_model.predict(X_test)
y_pred

"""# 3. Accuracy Score"""

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

"""# 4. Confusion Matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

import seaborn as sns
sns.heatmap(cm, annot=True)

"""# 5. Classification Report"""

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

"""# Thank You
---
"""
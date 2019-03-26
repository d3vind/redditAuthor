# Common import
import numpy as np
import pandas as pd
import itertools

# Set graph parameters
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as lemmatiser
# library for label encode
from sklearn.preprocessing import LabelEncoder
# libaries word clouds
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import pandas as pd

# Import libaries for tokenization and splitting set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# import TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
# import the multinomial version of naive bayes
from sklearn.naive_bayes import MultinomialNB
# import for measureing accuracy score
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix


# import for measureing accuracy score


df = pd.read_csv('outputfile.csv')


csv_path = '/home/dev/Documents/NLP/redditAuthor/outputfile.csv'
# read in our path
train = pd.read_csv(csv_path)
# print(train.head())

# Show word cloud
text = df.description[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(train['body'], train['author'], test_size=0.20, random_state=42)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


count_vector = CountVectorizer(stop_words='english')
X_train_counts = count_vector.fit_transform(X_train)
X_train_counts.shape, X_train.shape

# Tfidtransformation
# Term Frequency times Inverse Document Frequency

tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
print(X_train_tfidf[0])


# train a classifier on our data
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


# Prediction on test set
X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts)
X_test.shape, X_test_tfidf.shape


y_pred = classifier.predict(X_test_tfidf)

# check Accuracy
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Accuracy:", np.mean(y_pred == y_test))


# Detailed report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_mx = confusion_matrix(y_test, y_pred)
conf_mx





# # Importing necessary libraries
# Y = df['author']
#
# labelencoder = LabelEncoder()
# Y = labelencoder.fit_transform(Y)
#
#
# X = df['body']
#
# wordcloud1 = WordCloud().generate(X[0]) # for EAP
# wordcloud2 = WordCloud().generate(X[1]) # for HPL
# wordcloud3 = WordCloud().generate(X[3]) # for MWS
# print(X[0])
# print(df['author'][0])
# plt.imshow(wordcloud1, interpolation='bilinear')
# plt.show()
# print(X[1])
# print(df['author'][1])
# plt.imshow(wordcloud2, interpolation='bilinear')
# plt.show()
# print(X[2])
# print(df['author'][2])
# plt.imshow(wordcloud3, interpolation='bilinear')
# plt.show()

# Common import
import numpy as np
import pandas as pd
import itertools

# Set graph parameters
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# read in our path
csv_path = '/home/dev/Documents/NLP/redditAuthor/outputfile.csv'
train = pd.read_csv(csv_path)
# print(train.head())

greypo = train.loc[train['author'] == 'Greypo', 'body']
srbistan = train.loc[train['author'] == 'srbistan', 'body']
vc_wc = train.loc[train['author'] == 'vc_wc', 'body']

# #user counts
# print (vc_wc.count())
# print (greypo.count())
# print (srbistan.count())

# author string is defined to create the word cloud for each corresponding author
# def authorString(authorName):
#     frame = train.loc[train['author'] == authorName, 'body']
#     string = frame.to_string()
#     return string


# generate word clouds
# reference: https://www.datacamp.com/community/tutorials/wordcloud-python
# greyPoString = authorString("Greypo")
# srBistantoString = authorString("srbistan")
# vcwctoString = authorString("vc_wc")

#
#
# wordcloudU1 = WordCloud(collocations=False).generate(greyPoString)
# wordcloudU2 = WordCloud(collocations=False).generate(srBistantoString)
# wordcloudU3 = WordCloud(collocations=False).generate(vcwctoString)


# Display the generated image:
# plt.imshow(wordcloudU1, interpolation='bilinear')
# plt.show()
# plt.imshow(wordcloudU2, interpolation='bilinear')
# plt.show()
# plt.imshow(wordcloudU3, interpolation='bilinear')
# plt.show()
# plt.axis("off")

# split train and test sets from our large csv
X_train, X_test, y_train, y_test = train_test_split(train['body'], train['author'], test_size=0.20, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 80-20 splitting the dataset (80%->Training and 20%->Validation)

# tokenize and remove stop words
count_vector = CountVectorizer(stop_words='english')
X_train_counts = count_vector.fit_transform(X_train)
X_train_counts.shape, X_train.shape

# Tfidtransformation
# Term Frequency times Inverse Document Frequency
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
# print(X_train_tfidf[0])


RFclassifier = RandomForestClassifier()
RFclassifier.fit(X_train_counts, y_train)

# train a classifier on our data
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train_counts, y_train)


# Prediction on test set
X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts)
X_test.shape, X_test_tfidf.shape


y_predNB = NBclassifier.predict(X_test_counts)
y_predRF = RFclassifier.predict(X_test_counts)

# check Accuracy NB
print("Accuracy Score Test Naive Bayes:", accuracy_score(y_test, y_predNB))

print("Accuracy:", np.mean(y_predNB == y_test))

# check Accuracy RF
print("Accuracy Score Test Random Forest:", accuracy_score(y_test, y_predRF))
print("Accuracy:", np.mean(y_predRF == y_test))


# Detailed report for Naive Bayes
print(classification_report(y_test, y_predNB) + '\n')

# Detailed report for Random Forest
print(classification_report(y_test, y_predRF))


# https://scikit-learn.org/stable/
# auto_examples/applications/plot_face_recognition
# .html#sphx-glr-auto-examples-applications-plot-face-recognition-py


# labels = ['Greypo', 'srbistan', 'vc_wc']
# cm = confusion_matrix(y_test, y_predNB, labels)
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()




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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras import layers

filePath = '/home/dev/Documents/NLP/redditAuthor/outputfile.csv'

df = pd.read_csv(filePath, sep=',')

le = preprocessing.LabelEncoder()
df['author'] = le.fit_transform(df.author.values)
# print(df['author'].count)
authorEnc = df['author'].values
# df_author = df['author'].values
sentences = df['body'].values



sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, authorEnc, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)


input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

    # vectorizer = CountVectorizer()
    # vectorizer.fit(sentences_train)
    # X_train = vectorizer.transform(sentences_train)
    # X_test  = vectorizer.transform(sentences_test)
    #
    # classifier = LogisticRegression()
    # classifier.fit(X_train, y_train)
    # score = classifier.score(X_test, y_test)
    # print('Accuracy for {} data: {:.4f}'.format(source, score))

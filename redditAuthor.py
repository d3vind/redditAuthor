import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# library for label encode
from sklearn.preprocessing import LabelEncoder
# libaries word clouds
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('outputfile.csv')

print(df.to_string) # for showing a snapshot of the dataset
#
#
# Defining a module for Text Processing
def text_process(tex):

    # removes punctuation marks
    nopunct = [char for char in tex if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    # 2. Lemmatisation
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a = a+b+' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('english')]

# Importing necessary libraries
y = df['author']

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


X = df['body']
wordcloud1 = WordCloud().generate(X[0]) # for EAP
wordcloud2 = WordCloud().generate(X[1]) # for HPL
wordcloud3 = WordCloud().generate(X[3]) # for MWS
print(X[0])
print(df['author'][0])
plt.imshow(wordcloud1, interpolation='bilinear')
plt.show()
print(X[1])
print(df['author'][1])
plt.imshow(wordcloud2, interpolation='bilinear')
plt.show()
print(X[3])
print(df['author'][3])
plt.imshow(wordcloud3, interpolation='bilinear')
plt.show()

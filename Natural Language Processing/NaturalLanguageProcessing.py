'''Importing the Libraries'''
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

'''Downloading the Stopwords'''
'''try:
    print(nltk.data.find('orpus/stopwords'))
except LookupError:
    nltk.download('stopwords')'''
nltk.download('stopwords')

'''Importing the dataset'''
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
print(dataset.head())

'''Cleaning the Texts'''
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

'''Creating the Bag of words model'''
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

'''Splitting Dataset into Training and Test set'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

'''Fitting Naive Bayes Classifier to Training Set'''
classifier = GaussianNB()
classifier.fit(X_train, y_train)

'''Predicting the Test Set Results'''
y_pred = classifier.predict(X_test)
print(y_pred)

'''Making Confusion Matrix'''
cm = confusion_matrix(y_test, y_pred)
print(cm)

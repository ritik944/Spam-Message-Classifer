import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',
                      sep='\t',names=['label','message'])

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

ps= PorterStemmer()
corpus=[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating the bag of words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv= CountVectorizer(max_features=2500)
#x=cv.fit_transform(corpus).toarray()

#creating tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
vc=TfidfVectorizer()
x=vc.fit_transform(corpus).toarray()


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


#train test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

#traing model

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred=spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)

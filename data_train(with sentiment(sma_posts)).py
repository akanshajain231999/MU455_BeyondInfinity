
import pandas as pd

dataset = pd.read_csv('sma_posts.csv')
dataset = dataset.drop(['Id'], axis = 1)
dataset = dataset.dropna()
dataset.reset_index(inplace = True) 
dataset = dataset.drop(['index'], axis = 1) 

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 


corpus = []
count=0
#now do the same for every row in dataset. run to loop for all rows

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    lem = WordNetLemmatizer() #Another way of finding root word
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review]
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print("****************************************",count)
    count = count+1
    
"""**********************************************************************************************"""    
 #Adding corpus to csv 
#corpus_dataset = pd.DataFrame(corpus)
#corpus_dataset['corpus'] = corpus_dataset
#corpus_dataset = corpus_dataset.drop([0], axis = 1) 
#corpus_dataset.to_csv('corpus_dataset.csv')
#
#corpus = pd.read_csv('corpus_dataset.csv',engine='python')
#corpus = corpus['corpus'].tolist()
"""**********************************************************************************************"""    

"""**********************************************************************************************"""
#from sklearn.feature_extraction.text import TfidfVectorizer
#tf = TfidfVectorizer()
#features = tf.fit_transform(corpus).toarray()    
"""**********************************************************************************************"""
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = None)

features = cv.fit_transform(corpus).toarray()

"""**********************************************************************************************"""
def flag_df(dataset):
    
    if dataset['Sentiment'].upper()=="Positive".upper():
        
        return 1
    
    elif dataset['Sentiment'].upper()=="Neutral".upper():

        return 0
    
    else:

        return -1
    
#**************************************************************************
dataset['Sentiment1'] = dataset.apply(flag_df, axis = 1)
dataset = dataset.drop(['Sentiment'], axis = 1)

"""**********************************************************************************************"""

labels = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)

'''******************* linear regression***************************'''

from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

from sklearn.tree import DecisionTreeClassifier  #0.76
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)

from sklearn.ensemble import RandomForestClassifier  #0.81
classifier = RandomForestClassifier(n_estimators=50, random_state=0)  
classifier.fit(features_train, labels_train) 

from sklearn.svm import SVC # kernels: linear, poly and rbf
classifier = SVC(kernel = 'rbf', random_state = 0)  #0.71
classifier.fit(features_train, labels_train)


from sklearn.neighbors import KNeighborsClassifier  #0.69
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(features_train, labels_train)


from sklearn.naive_bayes import GaussianNB  #0.45
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

from sklearn.ensemble import ExtraTreesClassifier  #0.82
#from sklearn.datasets import make_classification
#X, y = make_classification(n_features=4, random_state=0)
classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)
classifier.fit(features_train, labels_train)

from sklearn.ensemble import GradientBoostingClassifier  #0.77
classifier = GradientBoostingClassifier(random_state=0)
classifier.fit(features_train, labels_train)

from sklearn.linear_model import LogisticRegression        #0.84
classifier = LogisticRegression(random_state=0).fit(features_train, labels_train)

from sklearn.naive_bayes import BernoulliNB  #0.69
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import GaussianNB  #0.45
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import MultinomialNB  #0.77
classifier = MultinomialNB()
classifier.fit(features_train, labels_train)


''' **************************************************************************'''

labels_pred = classifier.predict(features_test) 

compare = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (compare )

labels_test = labels_test.astype(int) #convert into integer value
labels_pred = labels_pred.astype(int) #convert into integer value
from sklearn.metrics import accuracy_score
accuracy_score(labels_test,labels_pred) 

''' **************************************************************************'''

input_data = ['Can it be delivered to Bangalore???'] 
  
input_data = cv.transform(input_data).toarray()

input_pred = classifier.predict(input_data)
input_pred = input_pred.astype(int)

if input_pred[0]==1:
    print("Positive")
elif input_pred[0]==0:
    print("Neutral")
else:
    print("Negative")






import pandas as pd
import codecs

dataset = pd.read_csv('train.csv')
dataset = dataset.dropna()
dataset.reset_index(inplace = True) 
dataset = dataset.drop(['index'], axis = 1) 

stopwords = codecs.open("stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')




#from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 


corpus = []
count=0


for i in range(0, len(dataset)):
    review = dataset['text'][i]
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords)]
    
    lem = WordNetLemmatizer() #Another way of finding root word
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review]
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    print("****************************************",count)
    count = count+1
    


"""from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
features = tf.fit_transform(corpus).toarray()    """
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
features = cv.fit_transform(corpus).toarray()

'''
import pickle
with open('CountVectorizer_multi.pkl','wb') as f:
    pickle.dump(cv,f)
'''


labels = dataset.iloc[:, 1].values




# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.30, random_state = 0)

'''******************* Classification***************************'''
'''
from sklearn.tree import DecisionTreeClassifier  #0.76    #0.57  -->0.654594
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)

from sklearn.ensemble import RandomForestClassifier  #0.81  #0.60  -->0.7016
classifier = RandomForestClassifier(n_estimators=25, random_state=0)  
classifier.fit(features_train, labels_train) 

from sklearn.svm import SVC # kernels: linear, poly and rbf
classifier = SVC(kernel = 'rbf', random_state = 0)  #0.71      #0.45
classifier.fit(features_train, labels_train)


from sklearn.neighbors import KNeighborsClassifier  #0.69   # 0.46
classifier = KNeighborsClassifier(n_neighbors = 50, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(features_train, labels_train)


from sklearn.ensemble import ExtraTreesClassifier  #0.82    #0.59
classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)
classifier.fit(features_train, labels_train)

from sklearn.ensemble import GradientBoostingClassifier  #0.77   #0.58
classifier = GradientBoostingClassifier(random_state=0)
classifier.fit(features_train, labels_train)
'''
from sklearn.linear_model import LogisticRegression        #0.84  #0.62     --> 0.7432
classifier = LogisticRegression(random_state=0).fit(features_train, labels_train)
'''
from sklearn.naive_bayes import BernoulliNB  #0.69   #0.60
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import GaussianNB  #0.45    #0.37
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

from sklearn.naive_bayes import MultinomialNB  #0.77   #0.58
classifier = MultinomialNB()
classifier.fit(features_train, labels_train)

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
classifier = GridSearchCV(LogisticRegression(solver='liblinear'), tuned_parameters, cv=5, scoring="accuracy")
classifier.fit(features_train, labels_train)
'''
'''
import pickle
with open('classifier_multi.pkl','wb') as f:
    pickle.dump(classifier,f)
'''

''' **************************************************************************'''

labels_pred = classifier.predict(features_test) 

compare = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (compare )

labels_test = labels_test.astype(int) #convert into integer value
labels_pred = labels_pred.astype(int) #convert into integer value
from sklearn.metrics import accuracy_score
accuracy_score(labels_test,labels_pred) 

''' **************************************************************************'''
'''
import pickle
with open('classifier_multi.pkl','rb') as f:
    classifier = pickle.load(f)
with open('CountVectorizer_multi.pkl','rb') as f:
    cv = pickle.load(f)
'''

''' *-********************************Predicting the Sentiment **************************************'''

import codecs

stopwords = codecs.open("stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')

input_data = ['we are going to win the tournament soon'] 
  #from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 

corpus = []
count=0
for i in range(0, len(input_data)):
    review = input_data[i]
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords)]
    
    lem = WordNetLemmatizer() #Another way of finding root word
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review]
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    print("****************************************",count)
    count = count+1
    
input_data = cv.transform(corpus).toarray()

input_pred = classifier.predict(input_data)
input_pred = input_pred.astype(int)

if input_pred[0]==1:
    print("Positive")
elif input_pred[0]==0:
    print("Neutral")
else:
    print("Negative")

''' *-********************************Deep Learning Model **************************************'''
''' ************************************************************************************************'''
import keras  #ANN
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


#adding the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5000,))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(features_train, labels_train, batch_size = 10, epochs = 10)

#accuracy while fitting --> 60.09, while predictng-->12.39


''' *****************************Plotting the graph*************************************'''


import wordcloud
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
word_cloud=WordCloud(width=1000,height=500,stopwords=STOPWORDS,background_color='white').generate(''.join(dataset['text']))
plt.figure(figsize=(15,8))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

##########################################################################

import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import clear_output
def plot_label_distribution(dataset):
    percentage = dataset.groupby('score').size() / len(dataset) * 100
    amount = dataset.groupby('score').size()

    fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=70)
    plt.tight_layout()

    dataset.groupby('score').count()['text'].plot(kind='pie', ax=axes[0], labels=['Negative ({0:.2f}%)'.format(percentage[-1]), 'Neutral ({0:.2f}%)'.format(percentage[0]), 'Positive ({0:.2f}%)'.format(percentage[1])])
    sns.countplot(x=dataset['score'], hue=dataset['score'], ax=axes[1])
    
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[1].set_xticklabels(['Negative ({})'.format(amount[-1]), 'Neutral ({})'.format(amount[0]), 'Positive ({})'.format(amount[1])])
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)

    axes[0].set_title('Target Distribution in Training Set', fontsize=13)
    axes[1].set_title('Target Count in Training Set', fontsize=13)

    plt.show()

plot_label_distribution(dataset)

#############################################################


import numpy as np   
methods = [ "DTC","RFC","SVC","KNN","GaussNB","ETC","GBC","LogR","BernNB","MultiNB"]
accuracy = [76,81,71,69,45,82,77,84,69,77]
colors = ["purple", "magenta","#CFC60E","#0FBBAE","red","blue","brown","grey","green","violet"]

sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.show()    



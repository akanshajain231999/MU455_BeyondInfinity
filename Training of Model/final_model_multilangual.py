
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


for i in range(0, len(dataset)):
    review = dataset['text'][i]
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords)]
    
    lem = WordNetLemmatizer() 
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
    

    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
features = cv.fit_transform(corpus).toarray()

'''
import pickle
with open('CountVectorizer_multi.pkl','wb') as f:
    pickle.dump(cv,f)
'''


labels = dataset.iloc[:, 1].values



'''******************* Classification***************************'''

from sklearn.linear_model import LogisticRegression  
classifier = LogisticRegression(random_state=0).fit(features, labels)

'''
import pickle
with open('classifier_multi.pkl','wb') as f:
    pickle.dump(classifier,f)
'''



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



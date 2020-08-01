import pandas as pd
import numpy as np
dataset = pd.read_csv("review_train.csv")

    
def flag_df(dataset):

    if 1<dataset['Score']<=5 and dataset['Sentiment']==1:
    
        return dataset['Score']
    
    elif dataset['Sentiment']==0:
        sc= -2*dataset['Score']
        return (int(sc)+dataset['Score'])
    
    else:
    
        return 0

dataset['Sentiment_Score'] = dataset.apply(flag_df, axis = 1)

review_test = dataset.drop(['Score','Sentiment'], axis = 1) 

review_train = dataset.drop(['Score','Sentiment'], axis = 1) 

final = pd.concat([review_test, review_train])

final.reset_index(inplace = True) 
final = final.drop(['index'], axis = 1) 

final.to_csv('review_train_test.csv')
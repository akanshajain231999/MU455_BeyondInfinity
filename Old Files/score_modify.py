import pandas as pd

dataset = pd.read_csv("dataframe_score1.csv")
    
def flag_df(dataset):
    
    if 0.1<dataset['Sentiment_Score']:
        
        return 1
    
    elif -0.1<dataset['Sentiment_Score']<=0.1:

        return 0
    
    else:

        return -1

dataset['Sentiment'] = dataset.apply(flag_df, axis = 1)

review_test = dataset.drop(['Sentiment_Score'], axis = 1) 

review_test.to_csv('final_train2.csv')
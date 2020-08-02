from defination import *
import pandas as pd
data = pd.read_csv("HindiSentiWordnet.txt", delimiter=' ')
data = data['LIST_OF_WORDS'].tolist()
comment = []
count=0
for i in data:
    trans_text = translation_to_eng(i)
    text_token = tokenization(trans_text)

    text_pos = pos_mark(text_token)
    
    wn_text_senti_score = senti_score(text_pos)
    
    vander_text_senti_score = vader_senti_score(trans_text)
    
    adjusted_score1 = adjusted_score(wn_text_senti_score,vander_text_senti_score)
    #print(adjusted_score1)  
    
    senti = Sentiment_value(adjusted_score1)
    comment.append(senti)
    print("**********************************",count)
    count+=1
    
def flag(score):
    
    if score.upper()=="Positive".upper():
        
        return 1
    
    elif score.upper()=="Neutral".upper():

        return 0
    
    else:

        return -1
    
back = comment
temp=[]
for i in comment:
    temp.append(flag(i))
    
c=[]
c = pd.DataFrame()
c = pd.DataFrame(temp)
c['score'] = c
c = c.drop([0], axis = 1) 

d=[]
d = pd.DataFrame()
d = pd.DataFrame(data)
d['text'] = d
d = d.drop([0], axis = 1) 

final_train_hindi = pd.concat([d,c],axis=1)
final_train_hindi.reset_index(inplace = True) 
final_train_hindi = final_train_hindi.drop(['index'], axis = 1) 


final_train_hindi.to_csv('HindiSentiWordnet.csv',encoding='utf-8')


import pandas as pd
df = pd.read_csv("final_hindi.csv")
df1 = pd.read_csv("sample.csv")
df2 = pd.read_csv("HindiSentiWordnet1150-1800.csv")
df3 = pd.read_csv("HindiSentiWordnet1800-2400.csv")
df4 = pd.read_csv("HindiSentiWordnet.csv")

final_train = pd.concat([df,df1,df2,df3,df4],axis=0)
final_train.reset_index(inplace = True) 
final_train = final_train.drop(['index'], axis = 1) 
final_train.to_csv('HindiSentiWordnet_final.csv')
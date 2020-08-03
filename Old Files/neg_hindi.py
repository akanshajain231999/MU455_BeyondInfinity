import codecs
import pandas as pd
neg_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
data = []
score = []
sc= 1
for line in neg_reviews.split('$'):
    data.append(line.strip())
for i in range(len(data)):
    score.append(sc)
    
data = pd.DataFrame(data)
data['comment_text'] = data
data = data.drop([0], axis = 1) 

score = pd.DataFrame(score)
data['score'] = score


data.to_csv('pos_hindi.csv')
import pandas as pd
dataset = pd.read_csv('final_train_hindi.csv')
dataset = dataset.dropna()
data = pd.read_csv('neg_hindi.csv')
data = data.dropna()

final_train_hindi = pd.concat([dataset,data],axis=0)
final_train_hindi.reset_index(inplace = True) 
final_train_hindi = final_train_hindi.drop(['index'], axis = 1) 
final_train_hindi.to_csv('final_train_hindi.csv')

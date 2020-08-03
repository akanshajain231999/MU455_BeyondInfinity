import pandas as pd
data = pd.read_csv("positive-words.txt", delimiter=' ')
dataset = pd.read_csv("negative-words.txt", delimiter=' ')
data = data['text'].tolist()
text = []
score = []
sc = 1
for line in data:
    text.append(line.strip())
for i in range(len(data)):
    score.append(sc)
    
text = pd.DataFrame(text)
text['text'] = text
text = text.drop([0], axis = 1) 

score = pd.DataFrame(score)
text['score'] = score

final = pd.concat([text,text1],axis=0)
final.reset_index(inplace = True) 
final = final.drop(['index'], axis = 1) 
final.to_csv('final_train_hindi.csv')
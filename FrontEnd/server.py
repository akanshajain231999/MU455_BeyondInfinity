from flask import Flask, render_template, request,jsonify
app = Flask(__name__)
import pandas as pd
import codecs

stopwords = codecs.open("stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')

import pickle
with open('classifier_multi.pkl','rb') as f:
    classifier = pickle.load(f)
with open('CountVectorizer_multi.pkl','rb') as f:
    cv = pickle.load(f)


@app.route("/")
def view_template():
    return render_template("index.html")

@app.route("/data", methods=["GET","POST"])
def form_data():
    if request.method == "GET":
        return "<h1>Sorry, You mistaken somewhere</h1>"
    else:
        user_data = request.form   
        selected = user_data['selected']
        
        if int(selected)==1:
            text = user_data["text_area"]
            
            text = str(text).strip()
            print(text)
            input_data = [] 
            input_data.append(text)
            input_data = cv.transform(input_data).toarray()

            input_pred = classifier.predict(input_data)
            input_pred = input_pred.astype(int)
            
            if input_pred[0]==1:
                result = "Positive"
            elif input_pred[0]==0:
                result = "Neutral"
            else:
                result = "Negative"
            #print(result)            
            return jsonify(msg=str(result))
        
        if int(selected)==0:
            imge = user_data["img_name"]
            print(type(imge))
            #imge = "insert_file.csv"
            
            dataset = pd.read_csv(imge,engine = "python")
            data = dataset['text'].tolist()
            score=[]
            for item in data:
                 text = str(item).strip()
                 #print(text)
                 input_data = [] 
                 input_data.append(text)
                 from nltk.stem.wordnet import WordNetLemmatizer 
                
                 corpus = []
                 count=0
 
                 for i in range(0, len(input_data)):
                     review = input_data[i]
                     review = review.lower()
                     review = review.split()
                     review = [word for word in review if not word in set(stopwords)]
                    
                     lem = WordNetLemmatizer() #Another way of finding root word
                  
                     review = [lem.lemmatize(word) for word in review if not word in set(stopwords)]
                     review = ' '.join(review)
                     corpus.append(review)
                     print("****************************************",count)
                     count = count+1
                 input_data = cv.transform(corpus).toarray()
                 input_pred = classifier.predict(input_data)
                 input_pred = input_pred.astype(int)
                 if input_pred[0]==1:
                     sc = "Positive"
                 elif input_pred[0]==0:
                     sc = "Neutral"
                 else:
                     sc = "Negative"
                 score.append(sc)
            result = ', '.join(score)
            print(score)
            result_test = pd.DataFrame(score)
            result_test.to_csv('result_test.csv')
            return jsonify(msg=str(result))
            


if __name__ == "__main__":
  app.run(debug=True,use_reloader=False)
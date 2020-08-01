Smart India Hackathon 2020

....................................................................................................................................................................
Problem Statement-
A complete software web application which provide standard solution handle multiple code-mixed Indian languages English, Hindi, Kannada ,Bengali, Urdu etc and perform context aware Sentiment Analysis give results such as “Positive”, “Negative“ and “Neutral”.


Problem Analysis
....................................................................................................................................................................

	Analysis on Different languages

	Identifying Need of Maintenance

	Homophonic Word

	Noise Detection

	Strain in Finding Meaning  

	Dataset without Sentiment


Requirements:
..............................................................................................................................................................................


	python3 (Anaconda environment is preferred)

	Scikit-learn for performing different classification model on dataset

	Pandas to read and write csv files 

	Numpy for performing different operations on dataset

	NLTK for text mining in dataset

	Googletrans for translating any mixed language to english

	Pickle to load the trained model for future use

	Codecs

	Matplotlib for visualisation



We have used three approaches to classify the sentiment of text reviews as positive or Negative:
.............................................................................................................................................................................


1.	Resource Based Semantic Analysis using HindiSentiWordnet.---> 
    In this approach we used Hindi Sentiwordnet to classify the review's sentiment.
		
2.	IN language Semantic Analysis. : 
    This approach is based on training the classifiers on the same language as text.
		
3.	Machine Translation Based Semantic Analysis. : 

    In this approach we train the classifier on English reviews and for testing, we translate the Hindi reviews or any other mixed language  into English using Googletrans api       and then we classify the Sentiment of review.



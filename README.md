# Document Classification

This is an implementation of document classification using four algorithms: Multinomial Naive Bayes, Support Vector Machines, Random Forests and Logistic Regression.

Validation Sore is calculated as ( #correct - #incorrect )/( total )  
SVM has shown to have the highest validation score.

## Requirements:  
1)Python 2 or 3  
2)sklearn python module  
3)nllk python module and download stop words beforehand if possible  
4)pickle python module  
5)sys python module  
6)time python module  
7)pathlib python module  

## Variables  

ALL of these variables can be found in the main() function

1) cachedStopWords = False - list of stop words to remove.  
   you can define your own stop words. if "False" it will download from NLTK and use stopwords provided from them.  
   cachedStopwords = stopwords.words("english")
2) training_file = "trainingdata.txt" - this is the path of training file.  
3) testing_file = "testingdata.txt" - this is the path of the testing file.  
4) save_model_file = "document_classifier.sav" - this is the file in which the model wll be saved in.  
5) save_vectorizer_file = "tfidf_vectorizer.sav" - path of the file to save the vectorizer model in (should be a .sav file)  
6) load_model_status = True - if true it will load the model from the file with the name stored in the variable load_model_from, if False it will create a new model.  
7) Load_model_from = "document_classifier.sav" - this is the path of the file from which the model will be loaded from.  
8) load_vectorizer_from = "tfidf_vectorizer.sav" - path of the file to load the vectorizer from. if the file does not exist, it will use    the tfidf vectoizer from scikit-learn by default.  
9) validation_size = 0.25 - this is the split ratio for training vs validation data.  
   -> size(validation_data) = validation_size * size(total_data)  
10) classifier_to_use = "svm" - this is the classifier to use.  
    Choose from: "Multinomialnb", "svm", "random_forest" and "logistic_regression" 

## functions
score() function that will calculate the score as 100*(#correct-#incorrect)/(T)

load_test_file() function to load the testing file data given the testing_file would be a certain format

help() function to display help
please note for this function to work you will need the help.txt (path stored in the variable help_file)


"document_classifier" is a class with the following member functions: __init__(), train(), predict() and save_model()

## To run  
'''
python document_classification.py
'''
''' 
python document_classification.py -h 
'''
''' 
python document_classification.py -help 
'''

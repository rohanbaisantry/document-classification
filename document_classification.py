
# Document classification

# imports
from __future__ import print_function
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import sys
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# imports for models
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# function that will calculate the score as 100*(#correct-#incorrect)/(T)
def scoring(predicted, actual):
	n_correct, n_incorrect = 0, 0
	for i in range(len(predicted)):
		if predicted[i] == actual[i]:
			n_correct += 1
		else:
			n_incorrect += 1
	score = 100 * float( (n_correct - n_incorrect)/len(predicted) )
	return score

# function to load the testing file data
def load_test_file(testing_file, cachedStopWords):
	i = 0
	x_test = []
	if cachedStopWords == False:
		cachedStopWords = stopwords.words("english")
	with open(testing_file, "r") as f:
			text = f.readlines()
			for n in (text):
				if i == 0:
					n_test = int(n) 
				else:
					text =' '.join(n.split()[0:])
					text = ' '.join([word for word in text.split() if word not in cachedStopWords])
					x_test.append(text)
				i += 1
	print("\n Test Data loaded from \"" + str(testing_file) + "\"")
	return x_test

# funcion for print contents of "help.txt"
def help(help_file):
	with open(help_file, "r") as f:
		print("\n\n", f.read(), "\n\n")
	sys.exit()

class document_classifier:

	def __init__(self, train_file_path, validation_size=0.2, load_model=False, load_model_from="", cachedStopWords = False, vectorizer = ""): 
		self.targets = []
		self.data = []
		self.load_model = load_model
		self.load_model_from = load_model_from
		if cachedStopWords == False:
			cachedStopWords = stopwords.words("english")
			print(" Stop words downloaded from NLTK module")
		i = 0
		with open(train_file_path, "r") as f:
			text = f.readlines()
			for n in (text):
				if i == 0:
					self.n_train = int(n)
				else:
					self.targets.append(int(n[0]))
					text =' '.join(n.split()[1:])
					text = ' '.join([word for word in text.split() if word not in cachedStopWords])
					self.data.append(text)
				i += 1
		if Path(vectorizer).is_file():
			self.vectorizer = pickle.load(open(vectorizer, 'rb'))
			print("\n Vectorizer loaded from \"" + str(vectorizer) + "\".")
		else:
			self.vectorizer = TfidfVectorizer()
			print("\n A Tfidf Vectorizer has been used.")
		self.data_tfidf = self.vectorizer.fit_transform(self.data)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_tfidf, self.targets, test_size=validation_size, random_state=42)
		self.n_train, self.n_validation = self.x_train.shape[0], self.x_test.shape[0]
		print("\n Data loaded and split into train and validation sets from \"" + str(train_file_path) + "\"")
		print("\n Size of training data: " + str(self.n_train) + "\n Size of validation data: " + str(self.n_validation))
		print("\n Object of class \"document_classification\" has been initialized.")

	def train(self, classifier_to_use="svm"):
		start = time.time()
		if self.load_model == True:
			self.model = pickle.load(open(self.load_model_from, 'rb'))
			print("\n Model loaded from \"" + self.load_model_from + "\".")
			self.model.fit(self.x_train, self.y_train)
		elif classifier_to_use.lower() == "multinomialnb":
			self.model = MultinomialNB()
			print("\n Model created using \"" + classifier_to_use + "\" classifer.")
			self.model.fit(self.x_train, self.y_train)
		elif classifier_to_use.lower() == "svm":
			self.model = svm.LinearSVC()
			print("\n Model created using \"" + classifier_to_use + "\" classifer.")
			self.model.fit(self.x_train, self.y_train)
		elif classifier_to_use.lower() == "random_forest":
			self.model = RandomForestClassifier()
			print("\n Model created using \"" + classifier_to_use + "\" classifer.")
			self.model.fit(self.x_train, self.y_train)
		elif classifier_to_use.lower() == "logistic_regression":
			self.model = LogisticRegression(random_state=0)
			print("\n Model created using \"" + classifier_to_use + "\" classifer.")
			self.model.fit(self.x_train, self.y_train)
		else:
			print("\n Please choose only one of following classifiers:\n\t multinomialnb\n\t svm\n\t random_forest\n\t logistic_regression")
			sys.exit()
		end = time.time()
		print("\n Training took ", end-start, "seconds.")
		validation_predictions = self.model.predict(self.x_test)
		self.training_score = scoring(validation_predictions, self.y_test)
		print("\n Training completed with a Validation Score of ", self.training_score, ".")

	def save_model(self, save_model_file, save_vectorizer_file):
		pickle.dump(self.model, open(save_model_file, 'wb'))
		pickle.dump(self.vectorizer, open(save_vectorizer_file, 'wb'))

	def predict(self, x_test):
		x_test_tfidf = self.vectorizer.transform(x_test)
		return self.model.predict(x_test_tfidf)

# main function
def main():
	start = time.time()

	help_file = "help.txt"
	if len(sys.argv) > 1:
		if sys.argv[1] == "-h" or sys.argv[1] == "-help":
			help(help_file)
		else:
			print("\n\n You can use -h or -help to get information about the program.\n\n")
			sys.exit()

	print("\n\n You can use -h or -help to get information about the program.\n\n\t\t START\n\n")

	cachedStopWords = False # list of stop words to remove.
	# you can define your own stop words. if "False" it will download from NLTK and use stopwords provided from them. 
	# cachedStopwords = stopwords.words("english")

	training_file = "trainingdata.txt" # path of training file
	testing_file = "testingdata.txt" # path of testing file
	save_model_file = "document_classifier.sav" # path of the file to save the model in (should be a .sav file)
	save_vectorizer_file = "tfidf_vectorizer.sav" # path of the file to save the vectorizer model in (should be a .sav file)
	load_model_status = True # if true, model is loaded. else it will be created.
	Load_model_from = "document_classifier.sav" # path of the file to load the model from (should be a .sav file)
	load_vectorizer_from = "tfidf_vectorizer.sav" # path of the file to load the vectorizer from. if the file does not exist, it will use the tfidf vectoizer by default.
	validation_size = 0.25 # split ratio for training vs validation data -> size(validation_data) = validation_size * size(total_data)
	classifier_to_use = "svm" # classifier to use
	# choose from: [ "Multinomialnb", "svm", "random_forest", "logistic_regression"]

	temp_object = document_classifier(train_file_path=training_file, validation_size=validation_size, load_model=load_model_status, load_model_from=Load_model_from, vectorizer=load_vectorizer_from)
	print("\n")
	temp_object.train(classifier_to_use=classifier_to_use)
	print("\n")

	x_test = load_test_file(testing_file=testing_file, cachedStopWords=cachedStopWords)
	predictions = temp_object.predict(x_test=x_test)
	print("\n predicted outputs (class numbers) are: ", predictions)
	print("\n")

	temp_object.save_model(save_model_file=save_model_file, save_vectorizer_file=save_vectorizer_file)
	print("\n Model is saved in \"" + save_model_file + "\".")
	print("\n Vectorizer is saved in \"" + save_vectorizer_file + "\".")


	end = time.time()
	print("\n\n The program ran for ", end-start, " seconds.")
	print("\n\n\n \t\t END\n\n")

# run
if __name__ == "__main__":
	main()

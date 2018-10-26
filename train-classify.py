"""Script reads from fft and mfcc files and trains using logistic regression & knn
"""

import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np

"""reads FFT-files and prepares X_train and y_train.
"""
def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.

		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that match genre-dir
		file_list = glob.glob(genre_dir)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)

	return np.array(X), np.array(y)


"""reads MFCC-files and prepares X_train and y_train.
"""
def read_ceps(genre_list, base_dir):
	X, y = [], []
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			y.append(label)

	return np.array(X), np.array(y)


def learn_and_classify(X_train, y_train, X_test, y_test, genre_list):

	print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	logistic_classifier = linear_model.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	print("logistic accuracy = " + str(logistic_accuracy))
	print("logistic_cm:")
	print(logistic_cm)

	knn_classifier = KNeighborsClassifier()
	knn_classifier.fit(X_train, y_train)
	knn_predictions = knn_classifier.predict(X_test)
	knn_accuracy = accuracy_score(y_test, knn_predictions)
	knn_cm = confusion_matrix(y_test, knn_predictions)
	print("knn accuracy = " + str(knn_accuracy))
	print("knn_cm:")
	print(knn_cm)

def main():
	# first command line argument is the base folder that consists of the fft files for each genre
	base_dir_fft  = sys.argv[1]
	# second command line argument is the base folder that consists of the mfcc files for each genre
	base_dir_mfcc = sys.argv[2]

	#genre_list1 = ["blues", "classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
	genre_list1 = ["country","metal","classical","jazz","pop","rock"]
	#genre_list2 = ["blues","country","disco","jazz","metal","pop","reggae","rock"]
	genre_list2 = ["country","metal","classical","jazz","pop","rock"]
	#use FFT
	X, y = read_fft(genre_list1, base_dir_fft)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)
	#print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	print('\n******USING FFT******')
	learn_and_classify(X_train, y_train, X_test, y_test, genre_list1)
	print('*********************\n')

	#use MFCC
	X, y = read_ceps(genre_list2, base_dir_mfcc)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
	print('******USING MFCC******')
	learn_and_classify(X_train, y_train, X_test, y_test, genre_list2)
	print('*********************\n')

if __name__ == "__main__":
	main()

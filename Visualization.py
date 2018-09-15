#### Functions to visualize data graphically ####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def visualiseCCD(X,c,i):
	sns.distplot(X);
	plt.title("CCD for class "+str(c)+" and feature "+str(i))
	plt.show()

def visualizeData(X):
	# if input is both X and Y, make Y as last column of X
	N = X.shape[0]	
	# sorting according to Y
	X = X[X[:,X.shape[1]-1].argsort()]
	# slicing matrix acc. to classes
	sliced_matrix = {}
	x = X[0][len(X[0])-1]
	last_index = 0
	for i in range(1, X.shape[0]):
		elem = X[i]
		q = elem[len(elem)-1]
		if q!= x:
			sliced_matrix[x] = X[last_index:i, :X.shape[1]-1]
			last_index = i
		x = q
	sliced_matrix[x] = X[last_index:X.shape[0], :X.shape[1]-1]

	for label in sliced_matrix:
		mat = sliced_matrix[label]
		for i in range(mat.shape[1]):
			# print(i)
			visualiseCCD(mat[:,i],label,i)

def visualizeConfusion(X):
	ax = sns.heatmap(X, annot=True, fmt="d", cbar = False)

	plt.title('Confusion matrix')
	plt.ylabel('True Class')
	plt.xlabel('Predicted Class')
	plt.show()

def visualizeCorrelation(cor_dict):

	for label in cor_dict:
		x = cor_dict[label]
		ax = sns.heatmap(x)
		plt.title('Correlation matrix for class '+str(int(label)))
		plt.ylabel('Features')
		plt.xlabel('Features')
		plt.show()

	



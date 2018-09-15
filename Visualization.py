#### Functions to visualize data graphically ####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def visualiseCCD(X,c,i):
	sns.distplot(X);
	plt.title("CCD for class "+str(c)+" and feature "+str(i))
	plt.show()

def visualizeDataCCD(X):
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

def visualizeKMeans(data,labelDict,k):
	colors = ("red", "green", "blue")
	groups = ("Class 0", "Class 1", "Class 2") 
	groupedData = []
	for label in labelDict:
		indices = labelDict[label]
		groupedData.append(data[indices,:].transpose())

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
	ax = fig.gca(projection='3d')

	for d, c, g in zip(groupedData, colors, groups):
		x, y, z = d
		ax.scatter(x, y, z, alpha=0.8, c=c, edgecolors='none', s=30, label=g)

	plt.title('3d KNN scatter plot')
	plt.legend(loc=2)
	plt.show()

def visualizeDataPoints(X):
	# data is complete matrix with labels

	colors = ("red", "green", "blue")
	groups = ("Class 0", "Class 1", "Class 2") 

	# X is complete data matrix with the label
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

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
	ax = fig.gca(projection='3d')

	i=0
	for label in sliced_matrix:
		mat_trans = sliced_matrix[label].transpose()
		ax.scatter(mat_trans[0], mat_trans[1], mat_trans[2], alpha=0.8, c=colors[i], edgecolors='none', s=30, label=groups[i])
		i+=1



	plt.title('3d data scatter plot')
	plt.legend(loc=2)
	plt.show()


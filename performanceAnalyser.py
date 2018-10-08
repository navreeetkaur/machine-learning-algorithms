import numpy as np
import math

def calcAccuracyTotal(Ypred,Ytrue):
	tot = len(Ypred)
	correct = 0
	for i in range(tot):
		if Ypred[i] == Ytrue[i]:
			correct+=1

	return correct*100.0/tot

def goodness(ytrue, ypred):
	n = len(ytrue)
	precision = {}
	recall = {}
	f1score = {}
	Classes, trues = np.unique(ytrue, return_counts = True)
	Classes1, predicts = np.unique(ypred, return_counts = True)
	for i in Classes:
		precision.update({i:0})
		recall.update({i:0})
	for i in range(0,n):
		if(ytrue[i] == ypred[i]):
			precision[ytrue[i]]+=1
			recall[ytrue[i]]+=1
	for i in range(0,len(trues)):
		precision[Classes[i]] = precision[Classes[i]]/predicts[i]
		recall[Classes1[i]] = recall[Classes1[i]]/trues[i]

	for label in precision:
		f1score[label] = 2 * precision[label]*recall[label]/(precision[label] + recall[label])
	# print(precision)
	# print(recall)
	return precision,recall,f1score


def getConfusionMatrix(ytrue,ypred):

	num_classes = len(np.unique(ytrue))
	confusion = np.zeros((num_classes,num_classes),dtype =np.int)

	for i in range(len(ytrue)):
		yt = int(ytrue[i])
		yp = int(ypred[i])
		# print(yt,yp)
		confusion[yt][yp] += 1

	return confusion

def getCorrelationMatrix(X):
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

	mu = {}
	sigma = {}
	correlation_dict = {}

	for label in sliced_matrix:
		# total data points of this class
		total_pnts = sliced_matrix[label].shape[0]
		# mu vector for this class
		mu[label] = np.asarray(sliced_matrix[label].sum(axis = 0))/(1.0*total_pnts)
		# subtract mu from each row for sigma computation. . . 
		sliced_matrix[label] = sliced_matrix[label] - mu[label]
		sigma[label] = np.matmul(sliced_matrix[label].transpose(),sliced_matrix[label])
		sigma[label] = sigma[label]*1.0/total_pnts	

		correlation_dict[label] = np.copy(sigma[label])

		for i in range(correlation_dict[label].shape[0]):
			for j in range(correlation_dict[label].shape[0]):
				correlation_dict[label][i][j] = sigma[label][i][j]/((sigma[label][i][i] ** 0.5)*(sigma[label][j][j] ** 0.5))

	return correlation_dict

def getFullCovariance(X):
	# X is only feature points
	total_pnts = X.shape[0]
	mu = np.asarray(X.sum(axis = 0))/(1.0*total_pnts)
	X = X - mu
	sigma = np.matmul(X.transpose(),X)
	sigma = sigma *1.0/total_pnts
	return sigma








import numpy as np
import math

class PerformanceCheck:
	def __init__(self):
		pass

	def calcAccuracyTotal(self,Ypred,Ytrue):
		tot = len(Ypred)
		correct = 0
		for i in range(tot):
			if Ypred[i] == Ytrue[i]:
				correct+=1

		return correct*100.0/tot

	def goodness(self,ytrue, ypred):
		n = len(ytrue)
		precision = {}
		recall = {}
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
		# print(precision)
		# print(recall)
		return precision,recall




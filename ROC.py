# This ROC file contains an Roc class
# It has three functions apart from an initialisation
# Among the initialisations, ytrue is a list of true class values
# yprob is a dictionary corresponding to each data point.
# The dictionary contains keys as the classes with values as the corresponding probabilities.
# The threshold is an np array which will specify the different threshold values to mark points



import numpy as np
import matplotlib.pyplot as plt
class Roc:

	def __init__(self,ytrue,yprob,threshold):
		self.ytrue = ytrue
		self.yprob = yprob
		self.threshold = threshold # an array of chosen threshold values for plotting


# The following functioni will take a classname and calculate necessary values like TPR and TNR
	def generate_start(self,classname):
		n = len(self.ytrue)
		ytrue = self.ytrue
		yprob = self.yprob
		threshold = self.threshold
		m = len(threshold)
		tpr = np.zeros(m)
		tnr = np.zeros(m)
		for item in range(0,m):
			# print(threshold[item])
			positive = 0 
			negative = 0
			TP = 0
			TN = 0
			for i in range(0,n):
				if(ytrue[i]==classname):
					positive+=1
				else:
					negative+=1
			for i in range(0,n):
				# print(ytrue[i])
				dic = yprob[i]
				maxprob = 0.0
				maxclass = 0.0
				for keys in dic:
					# print(keys)
					if(dic[keys]>maxprob):
						maxprob = dic[keys]
						maxclass = keys
				if(ytrue[i]==classname and maxclass==classname and maxprob>=threshold[item]):
					# print("hello")
					TP+=1
				elif(ytrue[i]!=classname and maxclass!=classname):
					TN+=1
				elif(ytrue[i]!=classname and maxclass==classname and maxprob<threshold[item]):
					TN+=1
			# print(TP)
			TPR = ((float)(TP))/(float)(positive)
			TNR = 1.0 - ((float)(TN))/(float)(negative)
			tpr[item] = TPR
			tnr[item] = TNR
		return tpr,tnr

#This function only plots values accrding to given arrays x and y

	def plot(self,x,y,classname):
		plt.plot(x,y)
		plt.xlabel('True Negative Rate')
		plt.ylabel('True Positive Rate')
		plt.title("Class - "+(str)(classname))
		plt.show()

# The following function actually generates as many ROC curves as there are classes in the data

	def Roc_gen(self): # this function needs to be called for ROC generation
		a = set(self.ytrue)
		for item in a:
			y,x = self.generate_start(item)
			self.plot(x,y,item)





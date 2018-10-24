# This ROC file contains an Roc class
# It has three functions apart from an initialisation
# Among the initialisations, ytrue is a list of true class values
# yprob is a dictionary corresponding to each data point.
# The dictionary contains keys as the classes with values as the corresponding probabilities.
# The threshold is an np array which will specify the different threshold values to mark points



import numpy as np
import matplotlib.pyplot as plt
class Roc:

	def __init__(self,ytrue,yprob,threshold,model_name):
		self.ytrue = ytrue
		self.yprob = yprob
		self.threshold = threshold # an array of chosen threshold values for plotting
		self.model_name = model_name

# The following functioni will take a classname and calculate necessary values like TPR and TNR
	def generate_start(self,classname):
		n = len(self.ytrue)
		ytrue = self.ytrue
		yprob = self.yprob
		threshold = self.threshold
		m = len(threshold)
		tpr = np.zeros(m)
		tnr = np.zeros(m)
		fpr = np.zeros(m)
		for item in range(0,m):
			ytrue_new = []
			ypred_new = []
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

			for i in range(n):
				if (self.ytrue[i] == classname):
					ytrue_new.append(1)
				else:
					ytrue_new.append(0)

				proba_pred_class = yprob[i][classname]
				if (proba_pred_class>=threshold[item]):
					ypred_new.append(1)
				else:
					ypred_new.append(0)

			for i in range(n):
				if int(ytrue_new[i]) == 1:
					positive +=1
					if int(ypred_new[i]) == 1:
						TP += 1
				if int(ytrue_new[i]) == 0 :
					negative +=1
					if int(ypred_new[i])==0:
						TN += 1 

			TPR = ((float)(TP))/(float)(positive)
			TNR = 1.0 - ((float)(TN))/(float)(negative)
			FPR = 1 - TNR
			tpr[item] = TPR
			tnr[item] = TNR
			fpr[item] = FPR
		return tpr,tnr,fpr


#This function only plots values accrding to given arrays x and y
	def plot(self,x,y,classname):
		plt.plot(x,y,'r',label=self.model_name)
		# plt.plot([0.0,1.0],[0.0,1.0],'k', label='Random Guess')
		plt.xlabel('True Negative Rate')
		plt.ylabel('True Positive Rate')
		plt.title("Class - "+(str)(classname))
		plt.legend()
		plt.show()


# The following function actually generates as many ROC curves as there are classes in the data
	def Roc_gen(self): # this function needs to be called for ROC generation
		a = set(self.ytrue)
		for item in a:
			tpr, tnr, fpr = self.generate_start(item)
			# self.plot(fpr, tpr ,item)
			self.plot(tnr, tpr, item)




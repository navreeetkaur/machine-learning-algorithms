import inputReader
import performanceAnalyser
import numpy as np


class percep_2:
	def __init__(self,input_data,labels,test_data):
		self.input_data = input_data
		self.labels = labels
		a = len(input_data[0])
		self.weights = np.zeros(a)
		self.test_data = test_data

	def delta(self,instance,label):
		pred = self.weights.dot(instance)
		if(pred>0 and label == 1):
			a = np.zeros(len(instance))
			return a
		elif(pred<0 and label == 0):
			a = np.zeros(len(instance))
			return a
		elif(pred<=0 and label==1):
			return instance
		elif(pred>=0 and label==0):
			return (-instance)

	def process(self,k):
		n = len(self.input_data)
		for i in range(0,k):
			for j in range(0,n):
				instance = self.input_data[j]
				label = self.labels[j]
				change = self.delta(instance,label)
				self.weights = self.weights + change

	def ypred(self):
		a = self.test_data
		b = np.zeros(len(a))
		for i in range(0,len(a)):
			pred  = self.weights.dot(a[i])
			if(pred>0):
				b[i] = 1
			else:
				b[i] = 0
		return b



def main():
	inp = inputReader.InputReader('railwayBookingList.csv',2)
	training_data = inp.Train
	test_data = inp.Test[:,:-1]
	ytrue = inp.Test[:,-1]
	input_data = training_data[:,:-1]
	labels = training_data[:,-1]
	per = percep_2(input_data,labels,test_data)
	per.process(10)
	ypred = per.ypred()
	print(performanceAnalyser.calcAccuracyTotal(ypred,ytrue))

if __name__=="__main__":
	main()




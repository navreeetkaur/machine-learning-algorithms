import inputReader
import performanceAnalyser
import numpy as np


class percep_2:
	def __init__(self,input_data,labels):
		self.input_data = input_data
		self.labels = labels
		a = length(input_data[0])
		self.weights = np.zeros(a)

	def delta(self,instance,label):
		pred = self.weights.dot(instance)
		if(pred>0 and label = 1):
			a = np.zeros(length(instance))
			return a
		else if(pred<0 label = 0):
			a = np.zeros(length(instance))
			return a
		else if(pred<=0 and label=1):
			return instance
		else if(pred>=0 and label=0):
			return (-instance)

	def process(self,k):
		n = length(self.input_data)
		for i in range(0,k):
			for j in range(0,n):
				instance = self.input_data[j]
				label = self.labels[j]
				change = self.delta(instance,label)
				self.weights = self.weights + change

	def main():
		inp = inputReader.InputReader


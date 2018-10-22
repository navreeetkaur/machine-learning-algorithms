import inputReader
import performanceAnalyser
import numpy as np
import Visualization


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

class multi_perceptron:
	def __init__(self,input_data,labels,num_classes,test_data):
		self.input_data = input_data
		self.labels = labels
		self.num_classes = num_classes
		a = len(input_data[0])
		self.weights = np.zeros((num_classes,a))
		self.test_data = test_data

	def change_weights(self,instance,label,weight_instance,desired_label):
		pred = weight_instance.dot(instance)
		if(pred>0 and label==desired_label):
			return weight_instance
		elif(pred<0 and label!=desired_label):
			return weight_instance
		elif(pred<=0 and label==desired_label):
			a = weight_instance+instance
			return a
		elif(pred>=0 and label!=desired_label):
			a = weight_instance-instance
			return a

	def process(self,iterations):
		n = len(self.input_data)
		iterns = 0
		j = 0
		x = []
		y = []
		while (iterns<iterations):
			a = self.input_data[j]
			label = self.labels[j]
			for l in range(0,self.num_classes):
				desired_label = l
				instance = a
				weight_instance = self.weights[l]
				b = self.change_weights(instance,label,weight_instance,desired_label)
				if(np.array_equal(weight_instance,b)==False):
					# x.append(iterns)
					# ypred = self.pred()
					# ytrue = self.labels
					# accuracy  = self.calcAccuracyTotal(ypred,ytrue)
					# y.append(accuracy)
					# y.append(-self.loss(weight_instance,instance))
					iterns+=1
				self.weights[l] = b
			j = (j+1)%n
		# plt.plot(x,y)
		# plt.show()

	def pred(self):
		test_data = self.test_data
		n = len(test_data)
		output = np.zeros(n)
		for i in range(0,n):
			predictions = np.zeros(self.num_classes)
			for j in range(0,self.num_classes):
				predictions[j] = self.weights[j].dot(test_data[i])
			output[i] = np.argmax(predictions)
		return output

	# def loss(self):
	# 	sum=0
	# 	test_data = self.test_data
	# 	n = len(test_data)
	# 	for i in range(0,n):
	# 		for j in range(0,self.num_classes):
	# 			sum+=self.weights[j].dot(test_data[i])
	# 	return sum

	def calcAccuracyTotal(self,Ypred,Ytrue):
		tot = len(Ypred)
		correct = 0
		for i in range(tot):
			if Ypred[i] == Ytrue[i]:
				correct+=1
		return correct*100.0/tot



def main():
	x = ["Medical_data.csv","test_medical.csv"]
	inp = inputReader.InputReader(x,0)
	training_data = inp.Train
	test_data = inp.Test[:,:-1]
	ytrue = inp.Test[:,-1]
	input_data = training_data[:,:-1]
	labels = training_data[:,-1]
	per = multi_perceptron(input_data,labels,test_data)
	per.process(19)
	ypred = per.ypred()
	print(performanceAnalyser.calcAccuracyTotal(ypred,ytrue))

if __name__=="__main__":
	main()




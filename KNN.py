import numpy as np

class KNN:
	def __init__(k,training_set,testing_set,names):
		self.k = k
		self.training_set = training_set
		self.testing_set = testing_set
		self.labels = np.zeros(len(testing_set))
		self.names = names

	def distance(self,data1,data2):
		n = len(data1)
		dis = 0
		arr = data1-data2
		arr = np.square(arr)
		dis = sum(arr)
		dis = pow(dis,0.5)
		return dis

	def sortedarr_k(self,data1):
		k = self.k
		n = len(self.training_set)
		arr = np.zeros((2,n))
		for i in range(0,n):
			arr[0][i] = i
			arr[1][i] = self.distance(data1,self.training_set[i])
		ind = np.lexsort(arr[0],arr[1])
		arr1 = np.zeros(n)
		for i in range(0,len(ind)):
			arr1[i] = arr[0][ind[i]]
		if(k>=n):
			return arr1
		else:
			return arr1[0:k]

	def labelling(self,data,position):
		k = self.k
		arr = self.sortedarr_k(data)
		arr1 = np.zeros(k)
		for i in range(0,k):
			arr1[i] = self.names[arr[i]]
		u,ver  = np.unique(arr, return_counts = True)
		#print(u)
		#print(ver)
		a = 0
		for i in range(0,len(ver)):
			if(ver[a]<ver[i]):
				a = i
		self.labels[position] = u[a]

	def allocate(self):
		testing_set = self.testing_set
		for i in range(0, len(testing_set)):
			self.labelling(testing_set[i],i)







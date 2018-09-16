import numpy as np

class KNN:
	def __init__(self,k,training_set,testing_set,names,label_with_distance=False,mode):
		self.k = k
		self.mode = mode
		self.training_set = training_set
		self.testing_set = testing_set
		self.labels = np.zeros(len(testing_set))
		self.names = names
		self.label_with_distance = label_with_distance

	def distance(self, data1, data2):


		if self.mode == 0:
			n = len(data1)
			dis = 0
			for i in range(0,n):
				dis+=pow(data1[i]-data2[i],2)
			dis = pow(dis,0.5)
			return dis

		if self.mode == 1:
			n = len(data1)
			dis = 0
			for i in range(0,n):
				dis+=math.fabs(data1[i]-data2[i])
			return dis

		if self.mode == 2:
			n = len(data1)
			dis = 0
			for i in range(0,n):
				x = math.fabs(data1[i]-data2[i])
				if x > dis:
					dis=x
			return dis

	def sortedarr_k(self,data1):
		k = self.k
		n = len(self.training_set)
		arr = np.zeros((2,n))
		for i in range(0,n):
			arr[0][i] = i
			arr[1][i] = self.distance(data1,self.training_set[i])
		# print(arr[0])
		ind = np.lexsort((arr[0],arr[1]))
		arr1 = np.zeros((2,n))
		for i in range(0,len(ind)):
			arr1[0][i] = arr[0][ind[i]]
			arr1[1][i] = arr[1][ind[i]]
		# print(arr1[:,0:k])
		if(k>=n):
			return arr1
		else:
			return arr1[:,0:k]

	def labelling(self,data,position):
		k = self.k
		arr = self.sortedarr_k(data)
		# print(arr)
		arr1 = np.zeros(k)
		for i in range(0,k):
			arr1[i] = self.names[(int)(arr[0][i])]
		# print(arr1)
		u,ver  = np.unique(arr1, return_counts = True)
		# print(u)
		# print(ver)
		a = 0
		for i in range(0,len(ver)):
			if(ver[a]<ver[i]):
				a = i
		self.labels[position] = u[a]

	def distance_wise_labelling(self,data,position):
		k = self.k
		arr = self.sortedarr_k(data)
		# print(arr)
		ind = np.lexsort((arr[1],arr[0]))
		arr1 = np.zeros((2,k))
		for i in range(0,k):
			arr1[0][i] = arr[0][ind[i]]
			arr1[1][i] = arr[1][ind[i]]
		prev=0
		maxsum = 0
		maxclass = 0
		while(prev<k):
			cursum = 0 
			curr = prev+1
			while(curr<k and arr1[0][curr] == arr1[0][prev]):
				if(arr1[1][prev]==0):
					self.labels[position] = self.names[arr1[0][prev]]
					return
				cursum+=(1/(float)(arr1[1][prev]))
				curr+=1
				prev+=1
			if(arr1[1][prev] == 0):
				self.labels[position] = self.names[(int)(arr1[0][prev])]
				return
			cursum+=(1/(float)(arr1[1][prev]))
			if(maxsum<cursum):
				maxsum = cursum
				maxclass = arr1[0][prev]
			prev+=1
		self.labels[position] = self.names[(int)(maxclass)]


	def allocate(self):
		testing_set = self.testing_set
		for i in range(0, len(testing_set)):
			if(self.label_with_distance):
				self.distance_wise_labelling(testing_set[i],i)
			else:
				self.labelling(testing_set[i],i)


def main():
	file = open("test_arr.txt","r")
	test_arr = np.asarray([[float(x) for x in line.split()] for line in file])
	file.close()
	n = len(test_arr)
	a = (int)(2*n/3)
	experiment = KNN(3,test_arr[:a,:-1],test_arr[a:,:-1],test_arr[:,-1])
	experiment.allocate()
	print(experiment.labels)

if __name__=="__main__":
	main()







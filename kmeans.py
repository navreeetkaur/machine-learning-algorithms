import numpy as np
import random
class k_means:
	def __init__(self,k,inputs):
		self.k = k
		self.input = inputs
		n = len(inputs[0])
		self.means_arr = np.zeros((k,n))
		self.labels = {}
		for i in range(0,k):
			labelList=[]
			self.labels.update({i:labelList})

	def init_guess(self):
		a = len(self.input[0])
		arr = np.zeros((self.k,a))
		row_nos = len(self.input)
		rows = []
		j=0
		while(j<self.k):
			i = random.randint(0,row_nos-1)
			if(i not in rows):
				rows.append(i)
				j+=1
			else :
				continue
		for i in range(0,self.k):
			arr[i] = self.input[rows[i]]
		self.means_arr = arr

	def normalise(self):
		m = len(self.input)
		n = len(self.input[0])
		for i in range(0,n):
			maxs = max(self.input[:,i])
			mins = min(self.input[:,i])
			self.input[:,i] = (self.input[:,i]-mins)/(maxs-mins)

	def distance(self, data1, data2):
		n = len(data1)
		dis = 0
		for i in range(0,n):
			dis+=pow(data1[i]-data2[i],2)
		dis = pow(dis,0.5)
		return dis

	def allocate(self, data, position):
		dis = len(self.input[0])
		pos = 0
		n = self.k
		for i in range(0, n):
			a = self.distance(data, self.means_arr[i])
			if(a<dis):
				dis = a
				pos = i
		if(position in self.labels[pos]):
			return
		else:
			self.labels[pos].append(position)

	def allocation(self):
		for i in range(0,len(self.input)):
			self.allocate(self.input[i],i)

	def update(self):
		b = False
		for i in range(0, self.k):
			lista = self.labels[i]
			init_arr = self.means_arr[i]
			arr = np.zeros(len(self.input[0]))
			n = len(lista)
			if(n==0):
				continue
			for items in lista:
				arr+=self.input[items]
			arr = arr/n
			self.means_arr[i] = arr
			if(np.array_equal(arr,init_arr)):
				continue
			else:
				b=True
		return b

	def printarr(self):
		for i in range(0, len(self.input)):
				print (self.input[i])

	def printmeans(self):
		print(self.means_arr)

	def printlabels(self):
		print(self.labels)

	def apply(self):
		self.normalise()
		self.init_guess()
		b = True
		counter = 0
		while(b and counter<=1000000):
			self.allocation()
			b = self.update()
		self.printlabels()
		self.printmeans()
		print(self.rms())

	def find(self, i): 
		for j in range(0, self.k):
			if(i in self.labels[j]):
				return j
		return -1

	def rms(self):
		dis = 0
		print("**")
		for i in range(0,len(self.input)):
			a = self.find(i)
			print(a)
			dis+=self.distance(self.input[i],self.means_arr[a])
		print("**")
		return dis


def main():
	file = open("test_arr.txt","r")
	test_arr = np.asarray([[float(x) for x in line.split()] for line in file])
	file.close()
	experiment = k_means(2,test_arr)
	experiment.apply()

if __name__=="__main__":
	main()


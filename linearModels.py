import inputReader
import performanceAnalyser
import Visualization

import numpy as np 

class LinearModels:

	phiDict = {0: 'Projection', 1: 'Polynomial Basis'}

	def __init__(self,phiMode,maxDegree, isRegularized, lambd):
		self.phiMode = phiMode			# 0 : Projection ;; 1 : 1,x,x2
		self.lambd = lambd
		self.isRegularized = isRegularized
		self.W = []
		self.maxDegree = maxDegree

	def calcPhiXRiver(self,X):
		###### This is a design function; experiment with different datasets #################
		if self.phiMode == 0 :
			# Projection
			biasX = np.hstack((X,np.ones(X.shape[0]).reshape(-1,1)))
			return biasX
		if self.phiMode == 1:
			# Basis of 1,x,x2 till degree n-1
			n=4
			basis = []
			for i in range(1,self.maxDegree+1):
				basis.append(X**i)
			phiX = np.hstack(basis)
			phiX = np.hstack((phiX,np.ones(phiX.shape[0]).reshape(-1,1)))
			return phiX


	def calcW(self,phiX,Y):

		if self.isRegularized:
			phiTrans = phiX.transpose()
			phiTransPhiPlusLambdInv = np.linalg.inv(np.matmul(phiTrans,phiX) + self.lambd * np.identity(phiX.shape[1]))
			moore_penrose = np.matmul(phiTransPhiPlusLambdInv,phiTrans)
			W = np.matmul(moore_penrose,Y.reshape(-1,1))
			return W

		if not self.isRegularized:
			phiTrans = phiX.transpose()
			phiTransPhiInv = np.linalg.inv(np.matmul(phiTrans,phiX))
			moore_penrose = np.matmul(phiTransPhiInv,phiTrans)
			W = np.matmul(moore_penrose,Y.reshape(-1,1))
			return W

	def train(self,train_data):
		X = train_data[:,:-1]
		Y = train_data[:,-1]
		phiX = self.calcPhiXRiver(X)
		self.W = self.calcW(phiX,Y)


	def test(self,test_data,isRegress = False):
		X = test_data
		phiX = self.calcPhiXRiver(X)
		Ypred = np.matmul(self.W.transpose(),phiX.transpose())
		Ypred = Ypred[0]

		if isRegress:
			return Ypred
		if not isRegress:
			YpredClassify = (Ypred > 0.5).astype(np.int)
			return YpredClassify
	

class MultiClassLinear:

	phiDict = {0: 'Projection', 1: 'Polynomial Basis'}

	def __init__(self,phiMode,maxDegree, learnRate,isRegularized, lambd):
		self.phiMode = phiMode			# 0 : Projection ;; 1 : 1,x,x2
		self.learnRate = learnRate
		self.maxDegree = maxDegree
		self.parameters = []
		self.isRegularized = isRegularized
		self.lambd = lambd

	def train(self,train_data):
		sliced_matrix = Visualization.sliceMatrix(train_data)
		num_labels = len(sliced_matrix)
		features = train_data[:,:-1]
		yOneShot = np.zeros((features.shape[0],num_labels),dtype=np.int)
		for i in range(train_data.shape[0]):
			yOneShot[int(train_data[i][-1])] = 1
		parameters = np.random.rand(num_labels,features.shape[1])
		# print(parameters)
		self.parameters = self.performGradientDescent(parameters,features,yOneShot)

	def performGradientDescent(self,theta,X,Y):
		for k in range(5):
			# print(k)
			for i in range(X.shape[0]):
			# for i in range(10):
				z=0
				for j in range(theta.shape[0]):
					z+= X[i].dot(theta[j])
				print(z)
				if z==0:
					print(theta)
					exit()

				for j in range(theta.shape[0]):
					theta[j] = (1-self.learnRate*self.lambd)*theta[j] - self.learnRate* (Y[i][j] - (theta[j].dot(X[i]))/z)*X[i]
				# print(theta)
				
		# print(theta)
		return theta

	def test(self,test_data):
		Ypred = np.zeros(test_data.shape[0],)
		for i in range(test_data.shape[0]):
			vals = np.matmul(self.parameters,test_data[i].reshape(-1,1))
			vals = vals.transpose()[0]
			print(test_data[i])
			print(vals)
			exit()





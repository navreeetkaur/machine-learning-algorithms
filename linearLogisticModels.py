import inputReader
import performanceAnalyser
import Visualization
import math

import numpy as np 

def calcPhiX(X,maxDegree):
	basis = []
	for i in range(1,maxDegree+1):
		basis.append(X**i)
	phiX = np.hstack(basis)
	phiX = np.hstack((phiX,np.ones(phiX.shape[0]).reshape(-1,1)))
	return phiX

class LinearModels:

	def __init__(self,maxDegree, isRegularized, lambd):
		self.lambd = lambd
		self.isRegularized = isRegularized
		self.W = []
		self.maxDegree = maxDegree

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
		phiX = calcPhiX(X,self.maxDegree)
		self.W = self.calcW(phiX,Y)


	def test(self,test_data,isRegress = False):
		X = test_data
		phiX = calcPhiX(X,self.maxDegree)
		Ypred = np.matmul(self.W.transpose(),phiX.transpose())
		Ypred = Ypred[0]

		if isRegress:
			return Ypred
		if not isRegress:
			YpredClassify = (Ypred > 0.5).astype(np.int)
			return YpredClassify
	

class MultiClassLinear:

	def __init__(self,maxDegree, learnRate,isRegularized, lambd):
		self.learnRate = learnRate
		self.maxDegree = maxDegree
		self.parameters = []
		self.isRegularized = isRegularized
		self.lambd = lambd

	# def train(self,train_data):
	# 	phiX = calcPhiX(train_data[:,:-1],self.maxDegree)
	# 	sliced_matrix = Visualization.sliceMatrix(train_data)
	# 	num_labels = len(sliced_matrix)
	# 	yOneShot = np.zeros((phiX.shape[0],num_labels),dtype=np.int)
	# 	for i in range(train_data.shape[0]):
	# 		yOneShot[i][int(train_data[i][-1])] = 1
	# 	# print(yOneShot)
	# 	# exit()

	# 	parameters = np.random.rand(num_labels,phiX.shape[1])
	# 	# print(parameters)
	# 	# print(phiX)
	# 	# print(parameters)
	# 	self.parameters = self.performGradientDescent(parameters,phiX,yOneShot)

	def train(self,train_data):
		phiX = calcPhiX(train_data[:,:-1],self.maxDegree)
		sliced_matrix = Visualization.sliceMatrix(train_data)
		num_labels = len(sliced_matrix)
		yOneShot = np.zeros((phiX.shape[0],num_labels),dtype=np.int)
		for i in range(train_data.shape[0]):
			yOneShot[i][int(train_data[i][-1])] = 1

		self.parameters = self.calcW(phiX,yOneShot)
		# print(self.parameters)

	def calcW(self,phiX,Y):
		phiTrans = phiX.transpose()
		if self.isRegularized:
			phiTransPhiPlusLambdInv = np.linalg.inv(np.matmul(phiTrans,phiX) + self.lambd * np.identity(phiX.shape[1]))
		else:
			phiTransPhiPlusLambdInv = np.linalg.inv(np.matmul(phiTrans,phiX))
		moore_penrose = np.matmul(phiTransPhiPlusLambdInv,phiTrans)
		W = np.matmul(moore_penrose,Y)
		return W.transpose()


	# def performGradientDescent(self,theta,X,Y):
	# 	for k in range(1):
	# 		print(k)
	# 		# print(X.shape[0])
	# 		for i in range(X.shape[0]):
	# 			z=0
	# 			for j in range(theta.shape[0]):
	# 				z+= X[i].dot(theta[j])
	# 			# print(z)
	# 			if z==0:
	# 				print("z=0 at i "+str(i))
	# 				print(X[i])
	# 				print(theta)
	# 				exit()

	# 			for j in range(theta.shape[0]):
	# 				if self.isRegularized:
	# 					theta[j] = (1-self.learnRate*(self.lambd/X.shape[0]))*theta[j] + self.learnRate* (Y[i][j] - (theta[j].dot(X[i]))/z)*X[i]/z
	# 				else:
	# 					theta[j] = theta[j] + self.learnRate* (Y[i][j] - (theta[j].dot(X[i]))/z)*X[i]/z
	# 			# print(theta)
				
	# 	print(theta)
	# 	return theta

	def test(self,test_data):
		Ypred = np.zeros(test_data.shape[0])
		phiX = calcPhiX(test_data,self.maxDegree)
		for i in range(test_data.shape[0]):
			vals = np.matmul(self.parameters,phiX[i].reshape(-1,1))
			vals = vals.transpose()[0]
			Ypred[i] = np.argmax(vals)
		return Ypred

class LogisticModels:

	def __init__(self,maxDegree, learnRate,GDthreshold,isRegularized, lambd):
		self.learnRate = learnRate
		self.W = []
		self.maxDegree = maxDegree
		self.GDthreshold = GDthreshold
		self.isRegularized = isRegularized
		self.lambd = lambd

	def train(self,train_data):
		X = train_data[:,:-1]
		Y = train_data[:,-1]
		phiX = calcPhiX(X,self.maxDegree)
		self.W = self.calcW(phiX,Y)

	def calcW(self,phiX,Y):
		W = np.random.rand(phiX.shape[1])
		while True:
			# print(W)
			z = np.zeros(phiX.shape[1])
			for i in range(phiX.shape[0]):
				wTx = W.dot(phiX[i])
				# print(wTx)
				z += (Y[i] - (math.exp(-wTx)/(1+math.exp(-wTx))))*phiX[i]
				# z += (Y[i] - 1/(1+np.exp(-wTx)))*phiX[i]
			if self.isRegularized:
				correction = self.learnRate*z + self.learnRate*self.lambd*W
			else:
				correction = self.learnRate*z
			W = W -correction
			maxx = math.fabs(np.max(correction))
			# print(maxx)
			if  maxx < self.GDthreshold:
				return W

	def test(self,test_data):
		X = test_data
		phiX = calcPhiX(X,self.maxDegree)
		Ypred = np.zeros(phiX.shape[0])
		for i in range(phiX.shape[0]):
			x = phiX[i].dot(self.W)
			if x < 0:
				Ypred[i] = 1
		return Ypred


class MultiClassLogistic:
	
	def __init__(self,maxDegree, learnRate, GDthreshold,isRegularized, lambd):
		self.learnRate = learnRate
		self.maxDegree = maxDegree
		self.parameters = []
		self.isRegularized = isRegularized
		self.lambd = lambd
		self.GDthreshold = GDthreshold

	def train(self,train_data):
		phiX = calcPhiX(train_data[:,:-1],self.maxDegree)
		sliced_matrix = Visualization.sliceMatrix(train_data)
		num_labels = len(sliced_matrix)
		yOneShot = np.zeros((phiX.shape[0],num_labels),dtype=np.int)
		for i in range(train_data.shape[0]):
			yOneShot[i][int(train_data[i][-1])] = 1

		parameters = np.random.rand(num_labels,phiX.shape[1])
		self.parameters = self.performGradientDescent(parameters,phiX,yOneShot)

	def performGradientDescent(self,theta,X,Y):
		thetaNew = np.copy(theta)
		while True:
			# print(X.shape[0])
			for i in range(X.shape[0]):
				z=0
				for j in range(thetaNew.shape[0]):
					z+= np.exp(X[i].dot(thetaNew[j]))

				for j in range(thetaNew.shape[0]):
					if self.isRegularized:
						thetaNew[j] = (1-self.learnRate*(self.lambd/X.shape[0]))*thetaNew[j] + self.learnRate* (Y[i][j] - (thetaNew[j].dot(X[i]))/z)*X[i]/z
					else:
						thetaNew[j] = thetaNew[j] + self.learnRate* (Y[i][j] - np.exp(thetaNew[j].dot(X[i]))/z)*np.exp(thetaNew[j].dot(X[i]))*X[i]/z

			if np.max(np.absolute(thetaNew-theta)) < self.GDthreshold:
				return thetaNew
			else:
				theta = thetaNew


	def test(self,test_data):
		Ypred = np.zeros(test_data.shape[0])
		phiX = calcPhiX(test_data,self.maxDegree)
		for i in range(test_data.shape[0]):
			vals = np.matmul(self.parameters,phiX[i].reshape(-1,1))
			vals = vals.transpose()[0]
			Ypred[i] = np.argmax(vals)
		return Ypred
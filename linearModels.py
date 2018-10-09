import inputReader
import performanceAnalyser

import numpy as np 

class LinearModels:

	phiDict = {0: 'Projection'}

	def __init__(self,phiMode,isRegularized, lambd):
		self.phiMode = phiMode			# 0 : Projection
		self.lambd = lambd
		self.isRegularized = isRegularized
		self.W = []

	def calcPhiXRiver(self,X):
		###### This is a design function; experiment with different datasets #################
		if self.phiMode == 0 :
			biasX = np.hstack((X,np.ones(X.shape[0]).reshape(-1,1)))
			return biasX
		if self.phiMode == 1:
			basis = []
			n=4
			for i in range(n):
				basis.append(X**i)
			phiX = np.hstack(basis)
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
		# phiMode => 0 : Projection
		X = train_data[:,:-1]
		Y = train_data[:,-1]
		phiX = self.calcPhiXRiver(X)
		self.W = self.calcW(phiX,Y)


	def test(self,test_data):
		X = test_data
		phiX = self.calcPhiXRiver(X)
		Ypred = np.matmul(self.W.transpose(),phiX.transpose())
		return Ypred[0]
		

def main():
	inp = inputReader.InputReader('river_data.csv',mode=3,train_test_ratio = 0.8)
	# lm = LinearModels(1,True,10)
	lm = LinearModels(1,False,10e-2)
	lm.train(inp.Train)
	Ypred = lm.test(inp.Test[:,:-1])
	Ytrue = inp.Test[:,-1]
	print(lm.W)
	print(Ytrue)
	print(Ypred)
	rms = performanceAnalyser.calcRootMeanSquareRegression(Ypred,Ytrue)
	YtrueAvg = np.sum(Ytrue)/len(Ytrue)
	rmsPercent = rms*100/YtrueAvg
	print(rms,rmsPercent)


if __name__=="__main__":
	main()
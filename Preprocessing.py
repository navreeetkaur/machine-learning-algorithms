import numpy as np
import math
import operator
import sys


class Normalise(object):

	def __init__(self):
		self.scale_mean = []
		self.scale_sigma = []


	def scale(self, X, train = True):
		if train:
			d = X.shape[1] 
			n = X.shape[0]
			means = X.sum(axis=0)/(1.0*n)
			self.scale_mean = means
			# data with zero mean
			X = X - means
			# variance of each feature
			Y = np.square(X)
			Y = np.sqrt((Y.sum(axis=0))/(1.0*n))
			self.scale_sigma = Y
			# scaled data with zero mean and variance one
			X = np.divide(X,Y)
			return X
		elif len(self.scale_mean)!=0 and len(self.scale_sigma)!=0:
			X = np.divide((X - self.scale_mean),(1.0*self.scale_sigma))
			return X
		else:
			print("Scaling parameters not computed. Scale training data first !!")
			return



class PCA(object):
	def __init__(self, data, retain_var = 0.95, whiten = False, regularization = 10e-5):
		self.retain_var = retain_var
		self.whiten = whiten
		self.regularization = regularization
		self.scale_mean = 0
		self.scale_sigma = 0
		self.X = self.scale(data)
		self.U = self.compute_eigen()
		self.W = 0
		# self.mean_Z = 0
		# self.var_Z = 0
		self.eigvecs = 0
		self.eigvals = 0
		self.k = 0
		

	# feature scaling by Z-scoring 
	def scale(self, X):
		print("Need for scale")
		d = X.shape[1] 
		n = X.shape[0]
		means = X.sum(axis=0)/(1.0*n)
		self.scale_mean = means
		# data with zero mean
		X = X - means
		# variance of each feature
		Y = np.square(X)
		Y = np.sqrt((Y.sum(axis=0))/(1.0*n))
		self.scale_sigma = Y
		# scaled data with zero mean and variance one
		X = np.divide(X,Y)
		# print(means, self.mean)
		# print(self.scale_sigma[:10],Y[:10])
		return X

	def compute_eigen(self):
		# compute covariance
		# cov_X = np.zeros(shape=(self.X.shape[1], self.X.shape[1]), dtype = np.float)
		# for i in range(self.X.shape[1]):
		# 	for j in range(i, self.X.shape[1]):
		# 		sum = 0
		# 		for k in range(self.X.shape[0]):
		# 			sum += self.X[k][j]* self.X[k][i]
		# 		cov_X[i][j] = (sum*1.0)/(self.X.shape[0])
		# more efficient way of computing covariance
		cov_X = (np.matmul(self.X.transpose(), self.X))/(1.0 * self.X.shape[0])
		print("covariance computed. YAYAY")

		# diagonalise covariance matrix to obtain eigenvalues
		eigvals, eigvecs = np.linalg.eig(cov_X)

		# sort eigenvalues in decreasing order and obtain corresponding eigenvectors
		eigvals, eigvecs = zip(*sorted(zip(eigvals, eigvecs), reverse=True))
		self.eigvals = eigvals
		self.eigvecs = eigvecs
		eigvals = np.asarray(eigvals)
		eigvecs = np.asarray(eigvecs)
		print("Eigenvalues and vectors computed. YAYAY")

		# choose top k eigenvectors to form U - top k is determined by retaining 95% variance
		sum_n_eigvals = eigvals.sum()
		curr_sum = eigvals[0]
		for k in range(1,len(eigvals)):
			curr_sum += eigvals[k]
			var_retained = curr_sum/(1.0*sum_n_eigvals)
			if var_retained >= self.retain_var:
				break

		#U = eigvecs[:][:k+1] # (k x d)
		self.k = k
		print("U computed. YAYAY")
		return U


	def reduce(self, X):
		print("Starting to reduce")
		# scale X with previous parameters
		X = np.divide((X - self.scale_mean),(1.0*self.scale_sigma))

		if self.whiten:
			self.W = np.add(self.eigvals, self.regularization)**(-0.5)
		else:
			self.W = np.array(X.shape[1])
		Z = np.multiply(self.W, (np.matmul(self.U, X.transpose()))).transpose() # n x d

		return Z[:][:k+1]

		# obtain Z i.e. dimensionaly reduced feature matrix; Z = U transpose[k x d] * X[d x n]
		# Z = np.matmul(self.U, X.transpose()).transpose() # ((k x d) * (d x n)) = (n x k)
		# print("Z computed. YYAYA")
		# if self.whiten:
		# 	W = whitening(Z)
		# 	return W
		# else:
		# 	return Z
		
		
	def whitening(self, Z):
		# Whitening(optional) -  divide each feature by variance of that feature
		# regularisation added so that very small values do not blow up W 
		# W = np.multiply(Z, np.add(self.eigvals, self.regularization)**(-0.5))
		mean_Z = Z.sum(axis=0)/(1.0*Z.shape[0])
		self.zeromean_Z = Z - mean_Z
		cov_Z = (np.matmul(self.zeromean_Z.transpose(), self.zeromean_Z))/(1.0 * Z.shape[0])
		# regularisation added so that very small values do not blow up W 
		self.var_Z = np.sqrt(cov_Z.diagonal() + self.regularization)
		W = np.divide(Z, self.var_Z)
		return W


	def retrieve(self, Z):
		# to recover the original data
		if not whiten:
			recover_X = np.matmul(Z, self.U[:][:k+1])
			recover_X = np.multiply(recover_X, self.scale_sigma)
			recover_X = np.add(recover_X, self.scale_mean)
		else:
			recover_X = np.multiply(Z, (self.W)**(-1))
			recover_X = np.matmul(recover_X, self.U[:][:k+1])
			recover_X = np.multiply(recover_X, self.scale_sigma)
			recover_X = np.add(recover_X, self.scale_mean)

		# if not whiten:
		# 	recover_X = np.matmul(self.U.transpose(), Z.transpose()).transpose()
		# else:
		# 	# elementwise multiplication by variance of each feature
		# 	recover_Z = np.multiply(Z, self.var_Z)
		# 	# multiply by sigma, plus means  
		# 	recover_Z = np.multiply()
		# 	# recover_Z = 
		# 	recover_X = np.matmul(self.U.transpose(), recover_Z.transpose()).transpose()
		# return recover_X
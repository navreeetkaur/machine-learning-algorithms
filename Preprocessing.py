import numpy as np
import math
import operator
import sys

class PCA(object):
	def __init__(self, retain_var, data, whiten = True, regularization = 10e-5):
		self.retain_var = retain_var
		self.whiten = whiten
		self.regularization = regularization
		self.X = scale(data)
		self.U = 0
		self.Z = 0
		self.W = 0

	# feature scaling by Z-scoring 
	def scale(self, X):
		d = X.shape[1] 
		n = X.shape[0]
		means = X.sum(axis=0)/(1.0*n)
		# data with zero mean
		X = X - means
		# variance of each feature
		Y = X**2
		Y = (Y.sum(axis=0))/(1.0*n)
		# scaled data with zero mean and variance one
		X = X/Y
		return X


	def reduce(self):
	# compute covariance
	cov_X = np.zeros(shape=(X.shape[1], X.shape[1]), dtype = np.float)
	for i in range(X.shape[1]):
		for j in range(i, X.shape[1]):
			sum = 0
			for k in range(X.shape[0]):
				sum += X[k][j]*X[k][i]
			cov_X[i][j] = (sum*1.0)/(X.shape[0])

	# diagonalise covariance matrix to obtain eigenvalues
	eigvals, eigvecs = np.linalg.eig(cov_X)

	# sort eigenvalues in decreasing order and obtain corresponding eigenvectors
	eigvals, eigvecs = zip(*sorted(zip(eigvals, eigvecs), reverse=True))
	eigvals = list(eigvals)
	eigvecs = list(eigvecs)

	# choose top k eigenvectors to form U - top k is determined by retaining 95% variance
	sum_n_eigvals = eigvals.sum()
	curr_sum = eigvals[0]
	for k in range(1:len(eigvals)):
		curr_sum += eigvals[k]
		var_retained = curr_sum/(1.0*sum_n_eigvals)
		if var_retained >= retain_var:
			break

	self.U = eigvecs[:][:k+1] # (k x d)
	# obtain Z i.e. dimensionaly reduced feature matrix; Z = U transpose[k x d] * X[d x n]
	Z = np.matmul(self.U, self.X.transpose()).transpose() # ((k x d) * (d x n)) = (n x k)
	self.Z = Z
	if self.whiten:
		whiten()
		return self.W
	else:
		return self.Z


	def whiten(self):
		# Whitening(optional) -  divide each feature by variance of that feature
		mean_Z = self.Z.sum(axis=0)/(1.0*Z.shape[0])
		zeromean_Z = self.Z - mean_Z
		cov_Z = (np.matmul(zeromean_Z.transpose(), zeromean_Z))/(1.0 * self.Z.shape[0])
		# regularisation added so that very small values do not blow up W 
		var_Z = np.sqrt(cov_Z.diagonal() + self.regularization)
		self.W = np.divide(self.Z, var_Z)
		

	def retrieve(self):
		# to recover the original data
		recover_X = np.matmul(self.U.transpose(), self.Z.transpose()).transpose()
		return recover_X
		






	





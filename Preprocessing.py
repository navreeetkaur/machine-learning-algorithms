import numpy as np
import math
import operator
import sys

class PCA(object):
	def __init__(self, data, retain_var = 0.95, whiten = False, regularization = 10e-5):
		self.retain_var = retain_var
		self.whiten = whiten
		self.regularization = regularization
		self.X = self.scale(data)
		self.U = self.compute_eigen()
		self.scale_mean = 0
		self.scale_sigma = 0

	# feature scaling by Z-scoring 
	def scale(self, X):
		d = X.shape[1] 
		n = X.shape[0]
		means = X.sum(axis=0)/(1.0*n)
		self.scale_mean = means
		# data with zero mean
		X = X - means
		# variance of each feature
		Y = X**2
		Y = np.sqrt((Y.sum(axis=0))/(1.0*n))
		self.scale_sigma = Y
		# scaled data with zero mean and variance one
		X = X/Y
		return X

	def compute_eigen(self):
		# compute covariance
		cov_X = np.zeros(shape=(self.X.shape[1], self.X.shape[1]), dtype = np.float)
		for i in range(self.X.shape[1]):
			for j in range(i, self.X.shape[1]):
				sum = 0
				for k in range(self.X.shape[0]):
					sum += self.X[k][j]* self.X[k][i]
				cov_X[i][j] = (sum*1.0)/(self.X.shape[0])

		# diagonalise covariance matrix to obtain eigenvalues
		eigvals, eigvecs = np.linalg.eig(cov_X)

		# sort eigenvalues in decreasing order and obtain corresponding eigenvectors
		eigvals, eigvecs = zip(*sorted(zip(eigvals, eigvecs), reverse=True))
		eigvals = list(eigvals)
		eigvecs = list(eigvecs)

		# choose top k eigenvectors to form U - top k is determined by retaining 95% variance
		sum_n_eigvals = eigvals.sum()
		curr_sum = eigvals[0]
		for k in range(1,len(eigvals)):
			curr_sum += eigvals[k]
			var_retained = curr_sum/(1.0*sum_n_eigvals)
			if var_retained >= self.retain_var:
				break

		U = eigvecs[:][:k+1] # (k x d)
		return U


	def reduce(self, X):
		# scale X with previous parameters
		X = (X - self.scale_mean)/(1.0*self.scale_sigma)
		# obtain Z i.e. dimensionaly reduced feature matrix; Z = U transpose[k x d] * X[d x n]
		Z = np.matmul(self.U, X.transpose()).transpose() # ((k x d) * (d x n)) = (n x k)
		if self.whiten:
			W = whitening(Z)
			return W
		else:
			return Z
		

	def whitening(self, Z):
		# Whitening(optional) -  divide each feature by variance of that feature
		mean_Z = Z.sum(axis=0)/(1.0*Z.shape[0])
		zeromean_Z = Z - mean_Z
		cov_Z = (np.matmul(zeromean_Z.transpose(), zeromean_Z))/(1.0 * Z.shape[0])
		# regularisation added so that very small values do not blow up W 
		var_Z = np.sqrt(cov_Z.diagonal() + self.regularization)
		W = np.divide(Z, var_Z)
		return W


	def retrieve(self, Z):
		# to recover the original data
		if not whiten:
			recover_X = np.matmul(self.U.transpose(), Z.transpose()).transpose()
		else:
			# recover_Z = 
			recover_X = np.matmul(self.U.transpose(), recover_Z.transpose()).transpose()
		return recover_X
		






	





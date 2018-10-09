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

			# to account for zero variance
			# Y = np.add(Y, self.regularization)

			Y = np.sqrt((Y.sum(axis=0))/(1.0*n))
			self.scale_sigma = Y
			# scaled data with zero mean and variance one
			X = np.divide(X, Y, out=np.zeros_like(X), where=Y!=0)
			return X

		elif len(self.scale_mean)!=0 and len(self.scale_sigma)!=0:
			X = np.divide((X - self.scale_mean),(1.0*self.scale_sigma))
			return X

		else:
			return



class PCA(object):
	def __init__(self, data, k, whiten = False, regularization = 10e-5):
		self.whiten = whiten
		self.regularization = regularization
		self.scale_mean = 0
		self.scale_sigma = 0
		self.x_rot = 0
		self.W = 0
		self.eigvecs = 0
		self.eigvals = 0
		self.k = k
		self.var_retained = 0
		self.X = self.scale(data)
		self.U = self.compute_eigen()
		

	# feature scaling by Z-scoring 
	def scale(self, X):
		d = X.shape[1] 
		n = X.shape[0]
		means = X.sum(axis=0)/(1.0*n)
		self.scale_mean = means
		# data with zero mean
		X = X - means
		# variance of each feature
		Y = np.square(X)

		# to account for zero variance - regularise
		# Y = np.add(Y, self.regularization)

		Y = np.sqrt((Y.sum(axis=0))/(1.0*n))
		self.scale_sigma = Y
		# scaled data with zero mean and variance one
		X = np.divide(X, Y, out=np.zeros_like(X), where=Y!=0)
		return X


	def compute_eigen(self):
		cov_X = (np.matmul(self.X.transpose(), self.X))/(1.0 * self.X.shape[0])
		# print("covariance computed. YAYAY")

		# diagonalise covariance matrix to obtain eigenvalues
		eigvals, eigvecs = np.linalg.eig(cov_X)

		# sort eigenvalues in decreasing order and obtain corresponding eigenvectors
		eig_sort = eigvals.argsort()
		eigvals = eigvals[eig_sort[::-1]]
		eigvecs = eigvecs[eig_sort[::-1]]
		# print("Eigenvalues and vectors computed. YAYAY")

		# calculate variance retained by keeping k components
		# var_retained=0
		sum_n_eigvals = eigvals.sum()
		# curr_sum = eigvals[0]
		curr_sum = 0
		for i in range(0,self.k):
			curr_sum += eigvals[i]
		
		var_retained = curr_sum/(1.0*sum_n_eigvals)

		U = eigvecs
		self.var_retained = var_retained

		print (f'Number of dimensions retained: [ {self.k} ]')
		print (f'Variance retained: [ {var_retained} ]')
		
		# rotated version of X in the space with eigenvector basis
		self.x_rot = np.matmul(U.transpose(), self.X.transpose()).transpose() # (d x d, d x n).T  -> n x d
		# print("x_rot. YAYAY")
		# print("U computed. YAYAY")
		# U is the matrix of column eigenvectors in descending order of their corresponding eigenvalues
		return U


	def reduce(self, X, train):
		if train:
			# print("Starting to reduce training data")
			if self.whiten:
				self.cov_x_rot = (np.matmul(self.x_rot.transpose(), self.x_rot))/(1.0 * self.x_rot.shape[0])
				self.W = np.add(self.cov_x_rot.diagonal(), self.regularization)**(0.5)
				# Z = np.divide(x_rot, self.W, out = np.zeros_like(x_rot), where=self.W!=0)
				Z = np.divide(self.x_rot, self.W)
			else:
				Z = self.x_rot

		else:
			# print("Starting to reduce testing data")		
			# scale X with previous parameters
			X = np.divide((X - self.scale_mean),(1.0*self.scale_sigma))
			# rotate X
			x_rot =  np.matmul(self.U.transpose(), X.transpose()).transpose()
			if self.whiten:
				Z = np.divide(x_rot, self.W)
			else:
				Z = x_rot

		return Z[:,:self.k]
		

	def retrieve(self, X):
		# getting X back
		if self.whiten:
			x_rot = np.multiply(self.W[:k], X) # n x k
		else:
			x_rot = X # n x k

		# un-rotate
		X = np.matmul(self.U[:,:k], x_rot.transpose()).transpose()  # n x k
		# un-scale
		X = np.multiply(X, self.scale_sigma)
		X = np.add(X, self.scale_mean)
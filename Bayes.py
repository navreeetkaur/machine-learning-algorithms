import numpy as np
import math
import operator
import sys

import Distributions


# global var for distributions

class Bayes:
	def __init__(self, isNaive, distribution=[]):
		# last column should be Y
		self.priors = []
		self.isNaive = isNaive
		self.distribution = distribution
		self.parameters = []


	# algorithm
	def train(self,data):
		# assume distribution of ccd - for now assume Gaussian, which is the default
		# calculate MLE
		self.calculate_priors(data)
		if not self.isNaive:
			# Multivariate Gaussian; parameter is of size 1; contains u and sigma dicts
			self.ml_estimate(data)
			return
		if len(self.distribution) != data.shape[1] - 1:
			print("distribution array is not equal to number of n_features = "+str(data.shape[1]-1))
			exit()
		self.ml_estimate(data)


	# calculates prior of each class
	def calculate_priors(self,data):
		N = data.shape[0]
		priors = {}
		# counting the number of instances for each class
		for i in range(data.shape[0]):
			elem = data[i]
			q = elem[len(elem)-1]
			if q in priors:
				priors[q] += 1
			else:
				priors[q] = 1

		for key in priors:
			priors[key] = (priors[key]*1.0)/N
		self.priors =  priors


	# function to predict labels -  to be called from main
	def fit(self, test_X):
		# multiply likelihood to priors
		print(len(self.parameters))
		print(self.parameters)
		predicted_class = []
		for x in test_X:
			# x = x[:-1]
			posteriors = {}
			for c in self.priors:
				# c = int(c)
				if not self.isNaive:
					likelihood = Distributions.gaussian_multivar(x[:-1], self.parameters[0][0][c], self.parameters[0][1][c])
				else:
					likelihood = 1
					for i in range(len(self.distribution)):
						di = self.distribution[i]
						if di == 0:
							likelihood = likelihood*Distributions.gaussian(x[i],self.parameters[i][0][c][0],self.parameters[i][1][c][0][0]) 
						elif di == 1:
							likelihood = likelihood*self.parameters[i][c][x[i]]
						elif di == -1:
							continue

				posteriors[c] = likelihood*self.priors[c]


			predicted_class.append(max(posteriors.items(), key=operator.itemgetter(1))[0])
		return predicted_class


	def ml_estimate(self,data):
		# Returns list of parameters for each distribution type; parameter list of length same as distribution
		# probability distribution of x given theta(parameters)
		# guassian
		n_features = data.shape[1] - 1
		if not self.isNaive:
			# Fit a multivariate Gaussian in this case
			self.parameters.append(Distributions.gaussian_mle(data))
			return
		for i in range(len(self.distribution)):
			X = np.vstack((data[:,i],data[:,-1]))
			X = X.transpose()
			di = self.distribution[i]
			if di == 0:
				self.parameters.append(Distributions.gaussian_mle(X))
			elif di ==1:
				self.parameters.append(Distributions.multinomial_mle(X))
			elif di == -1:
				self.parameters.append(-1)

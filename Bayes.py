import numpy as np
import math
import operator
import sys

import Distributions


# global var for distributions
dists = {0:"Gaussian", 1:"Binomial", 2:"Bernoulli", 3:"Uniform", 4:"Exponential", 5:"Poisson"}

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
		# print(self.priors)
		if not self.isNaive:
			# Multivariate Gaussian; parameter is of size 1; contains u and sigma dicts
			self.ml_estimate(data)
			# print(self.parameters)

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
		predicted_class = []
		# print(self.parameters[0])
		# print(self.parameters[1])
		for x in test_X:
			posteriors = {}
			for c in self.priors:
				c = int(c)
				# print(c)
				if not self.isNaive:
					likelihood = Distributions.gaussian_multivar(x[:-1], self.parameters[0][0][c], self.parameters[0][1][c])
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
			mu, sigma = Distributions.gaussian_multivar_mle(data)
			self.parameters =  [[mu,sigma]]
			return

		if len(distribution) != n_features:
			print("distribution array is not correct")
			exit()

		# for i in range(len(distribution)):
		# 	dist = distribution[i]





		







		

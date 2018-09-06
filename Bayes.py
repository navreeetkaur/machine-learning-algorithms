import numpy as np
import math
import operator
import sys

import Distributions


# global var for distributions
dists = {0:"Gaussian", 1:"Binomial", 2:"Bernoulli", 3:"Uniform", 4:"Exponential", 5:"Poisson"}

class Bayes:
	def __init__(self, data, naive = False):
		# last column should be Y
		self.data = data
		self.naive = naive
		self.priors = self.calculate_priors()


	# calculates prior of each class
	def calculate_priors(self):
		N = self.data.shape[0]
		priors = {}
		# counting the number of instances for each class
		for i in range(self.data.shape[0]):
			elem = self.data[i]
			q = elem[len(elem)-1]
			if q in priors:
				priors[q] += 1
			else:
				priors[q] = 1

		for key in priors:
			priors[key] = (priors[key]*1.0)/N

		return priors

			
	# algorithm
	def train(self, distribution = []):
		# assume distribution of ccd - for now assume Gaussian, which is the default
		# calculate MLE
		mu, sigma = self.ml_estimate(distribution)
		parameters = [mu, sigma]
		return distribution, parameters


	# function to predict labels -  to be called from main
	def fit(self, test_X, distribution = []):
		# get parameters of likelihood
		distribution, parameters = self.train(distribution=distribution)
		# multiply likelihood to priors
		predicted_class = []
		for x in test_X:
			posteriors = {}
			for c in self.priors:
				likelihood = Distributions.gaussian(x[:-1], parameters[0][c], parameters[1][c])
				posteriors[c] = likelihood*self.priors[c]
			# vals = list(self.priors.values())
			# predicted_class.append(max(vals))
		# get class with maximum posterior 
			# print(posteriors)
			predicted_class.append(max(posteriors.items(), key=operator.itemgetter(1))[0])
		return predicted_class


	def ml_estimate(self, distribution = []):
		# probability distribution of x given theta(parameters)
		# guassian
		if distribution==[]:
			mu, sigma = Distributions.gaussian_mle(self.data, self.naive)
		return mu,sigma








		

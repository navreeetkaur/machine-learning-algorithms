import numpy as np
import math
import operator
import sys

import Distributions


# global var for distributions
dists = {0:"Gaussian", 1:"Binomial", 2:"Bernoulli", 3:"Uniform", 4:"Exponential", 5:"Poisson"}

class Bayes:
	def __init__(self, data, naive = False):
		self.priors = self.calculate_priors(labels)
		# last column should be Y
		self.data = data
		self.naive = naive

	# calculates prior of each class
	def calculate_priors(self):
		N = self.data.shape[0]
		priors = {}
		# counting the number of instances for each class
		for i in range(data.shape[0]):
			elem = data[i]
			q = elem[len(elem)-1]
			if q in priors.keys:
				priors[q] += 1
			else:
				priors[q] = 1

		for key, value in priors.items():
			priors[key] = value/N
			

	# algorithm
	def train(self, distribution = 0):
		# assume distribution of ccd - for now assume Gaussian, which is the default
		# calculate MLE
		mu, sigma = ml_estimate(distribution)
		parameters.append([mu, sigma])
		return distribution, parameters


	# function to predict labels -  to be called from main
	def fit(self, distribution = 0):
		# get parameters of likelihood
		distribution, parameters = train(distribution=distribution)
		# multiply likelihood to priors
		predicted_class = []
		for x in test_X:
			posteriors = {}
			for c in list(self.priors.keys()):
				likelihood = Distributions.guassian(x, parameters[0], parameters[1])
				posteriors[c] = likelihood*self.priors[c]
			# vals = list(self.priors.values())
			# predicted_class.append(max(vals))
		# get class with maximum posterior 
			predicted_class.append(max(predicted_class.items(), key=operator.itemgetter(1))[0])
		return predicted_class


	def ml_estimate(self, distribution = 0):
		# probability distribution of x given theta(parameters)
		# guassian
		if distribution==0:
			mu, sigma = Distributions.guassian_mle(features, self.naive)








		

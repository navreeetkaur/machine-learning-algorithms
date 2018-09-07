import numpy as np
import math
import operator as op

# x is a data point
def gaussian_multivar(x, mu, sigma):
	# x is a n dimensional vector ; mu is a n dimensional vector; sigma is n*n covariance matrix
	scale = (( math.fabs(np.linalg.det(sigma)) * ((2*math.pi)**x.shape[0]))**(-0.5))
	exp = (np.exp( np.matmul(np.matmul(-0.5 * np.transpose(x-mu), np.linalg.inv(sigma)),  (x-mu) )))
	return scale*exp

def gaussian(x, mu, sigma2):
	# x is a number; u is a number; sigma2 is a number(variance)
	scale = math.sqrt((2*math.pi* sigma2))*(-1)
	exp = np.exp((x-mu)*(1.0/(2*sigma2)))
	return scale*exp


# X is the whole feature vector consisting of ALL data points
# returns dictionaries mu, sigma  and list of labels 
def gaussian_mle(X):
	# if input is both X and Y, make Y as last column of X
	N = X.shape[0]	
	# sorting according to Y
	X = X[X[:,X.shape[1]-1].argsort()]
	# slicing matrix acc. to classes
	sliced_matrix = {}
	x = X[0][len(X[0])-1]
	last_index = 0
	for i in range(1, X.shape[0]):
	    elem = X[i]
	    q = elem[len(elem)-1]
	    if q!= x:
	        sliced_matrix[x] = X[last_index:i, :X.shape[1]-1]
	        last_index = i
	    x = q
	sliced_matrix[x] = X[last_index:X.shape[0], :X.shape[1]-1]

	mu = {}
	sigma = {}

	for label in sliced_matrix:
		# total data points of this class
		total_pnts = sliced_matrix[label].shape[0]
		# mu vector for this class
		mu[label] = np.asarray(sliced_matrix[label].sum(axis = 0))/(1.0*total_pnts)
		# subtract mu from each row for sigma computation. . . 
		sliced_matrix[label] = sliced_matrix[label] - mu[label]
		sigma[label] = np.matmul(sliced_matrix[label].transpose(),sliced_matrix[label])
		sigma[label] = sigma[label]*1.0/total_pnts	

	return [mu, sigma]





# def binomial( x, n, p):
# 	p = (ncr(n, x))*(p**x)((1-p)**(n-x))
# 	return p

# def binomial_mle(X):
# 	pass

# def bernoulli( x, p):
# 	p = (p**x)*((1-p)**(1-x))
# 	return p


# def uniform( a, b):
# 	p = 1//(b-a)
# 	return p


# def exponential( x, lamda):
# 	if x < 0:
# 		return 0
# 	else:
# 		p = lamda*(np.exp()**(-1*lamda*))


# def poisson( x, lamda):
# 	p = (math.exp(-1*lamda)*(lamda**x))/math.factorial(x)
# 	return p


# def ncr(n, r):
#     r = min(r, n-r)
#     numer = reduce(op.mul, xrange(n, n-r, -1), 1)
#     denom = reduce(op.mul, xrange(1, r+1), 1)
#     return numer/denom

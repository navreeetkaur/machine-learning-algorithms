import numpy as np 
import scipy 
from svmutil import *

from inputReader import InputReader 
import performanceAnalyser

import matplotlib.pyplot as plt


# prepare data for LIBSVM
def get_data(mode):
	if mode==0:
		inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
	elif mode==1:
		inputDataClass = InputReader(['fashion-mnist_train.csv', 'fashion-mnist_test.csv'],1)
	elif mode==2:
		inputDataClass = InputReader('railwayBookingList.csv',2)

	X_train = inputDataClass.Train
	X = X_train[:,:-1].tolist()
	Y = X_train[:,-1].tolist()
	X, X_min, X_max = scale_data(X, train=True)
	x_test = inputDataClass.Test
	x = x_test[:,:-1].tolist()
	y = x_test[:,-1].tolist()
	x ,_, _ = scale_data(x, train=False, X_min=X_min, X_max=X_max)

	return X, Y, x , y


def scale_data(X, train, X_min=None, X_max=None):
	X = np.asarray(X)
	if train:
		X_min = np.amin(X, axis=0)
		X_max = np.amax(X, axis=0)
	X = 2*np.divide(np.subtract(X,X_min), np.subtract(X_max, X_min)) - 1
	return X, X_min, X_max
	

def train(x, y, is_kernel, svm_type=0, kernel_type=0, degree=2, cost=1, tolerance=0.01, n_crossval=5):
	"""
	-s svm_type : set type of SVM (default 0)
		0 -- C-SVC		(multi-class classification)
		1 -- nu-SVC		(multi-class classification)
		2 -- one-class SVM
		3 -- epsilon-SVR	(regression)
		4 -- nu-SVR		(regression)
	-t kernel_type : set type of kernel function (default 2)
		0 -- linear: u'*v
		1 -- polynomial: (gamma*u'*v + coef0)^degree
		2 -- radial basis function: exp(-gamma*|u-v|^2)
		3 -- sigmoid: tanh(gamma*u'*v + coef0)
		4 -- precomputed kernel (kernel values in training_set_file)
	-d degree : set degree in kernel function (default 3)
	-g gamma : set gamma in kernel function (default 1/num_features)
	-r coef0 : set coef0 in kernel function (default 0)
	-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	-m cachesize : set cache memory size in MB (default 100)
	-e epsilon : set tolerance of termination criterion (default 0.001)
	-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	-v n: n-fold cross validation mode
	-q : quiet mode (no outputs)
	"""
	prob = svm_problem(y, x, iskernel = is_kernel)
	parameter = '-s' + str(svm_type) + '-t'+ str(kernel_type) + '-d' + str(degree) + '-c' + str(cost)
		+ '-e' + str(tolerance) + '-v' + str(n_crossval)
	param = svm_parameter(parameter)
	m = svm_train(prob, param)
	return m


def predict(m, x, y, probability_estimates=0):
	param = '-b' + str(probability_estimates)
	p_label, p_acc, p_val = svm_predict(y, x, m, param)
	ACC, MSE, SCC = evaluations(y, p_label)
	return p_label, p_acc, p_val, ACC, MSE, SCC


# compute kernel matrix for whole data
def precompute_kernel(kernel, x, b=1, sigma=1, a=1, theta=1, degree=2):
	x = np.asarray(x)
	kernel_matrix = []
	for i in range(x.shape[0]):
		K_i_j = [i]
		for j in range(x.shape[0]):
			K_i_j.append(kernel(kernel, x[i], x[j], b, sigma, a, theta, degree))
		kernel_matrix.append(K_i_j)
	return kernel_matrix


# compute kernel 
def kernel(k, x, z, b=1, sigma=1, a=1, theta=1, degree=2):
	if k =='0':
		return linear_kernel(x,z,b)
	elif k == '1':
		return gaussian_kernel(x,z,sigma)
	elif k == '2':
		return sigmoid_kernel(x,z,a,theta)
	elif k == '3':
		return polynomial_kernel(x,z,b,degree)
	

# types of kernels
def linear_kernel(x, y, b=1):
	return np.matmul(x,y.transpose()) + b


def gaussian_kernel(x, y, sigma=1):
	return np.exp(-(np.linalg.norm(x-y)/(2*(sigma**2))))


def sigmoid_kernel(x, y, a=1, theta=1):
	return numpy.tanh(a*np.matmul(x, y.transpose())+ theta)


def polynomial_kernel(x, y, b=1, degree=2):
	return (b + np.matmul(x, y.transpose()))**degree


def main(mode, kernel):
	X, Y, x, y = get_data(mode)
	if kernel==-1:
		is_kernel = False
	else:
		is_kernel = True
		X = precompute_kernel(kernel, X)#, b, sigma, a, theta, degree)
		x = precompute_kernel(kernel, x)#, b, sigma, a, theta, degree)

	model = train(X, Y, is_kernel)
	p_label, p_acc, p_val, ACC, MSE, SCC = predict(model, x, y)

	print(f'Accuracy: {ACC, p_acc}, MSE: {MSE}, p_val: {p_val}')


if __name__ == '__main__':
	mode = sys.argv[1]
	kernel = sys.argv[2]

	main(mode)




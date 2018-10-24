import numpy as np 
import scipy 
from src.inputReader import InputReader 
import src.performanceAnalyser as performanceAnalyser
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, './libsvm-3.23/python')
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score #y_true, y_pred
from svmutil import *
from svm import *


# prepare data for LIBSVM
def get_data(mode):
    if mode==0:
        inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
    elif mode==1:
        inputDataClass = InputReader(['fashion-mnist_train.csv', 'fashion-mnist_test.csv'],1)
    elif mode==2:
        inputDataClass = InputReader('railwayBookingList.csv',2)
    elif mode==3:
    	inputDataClass = InputReader('river_data.csv',3)
    else:
    	inputDataClass = None
    	print('INVALID MODE')
    	exit()

    X_train = inputDataClass.Train
    X_train = inputDataClass.Train
    x_test = inputDataClass.Test
    if mode==1:
        pca = PCA(n_components=80)
        X_new = pca.fit_transform(X_train[:,:-1])
        X_train = np.column_stack([X_new, X_train[:,-1]])
        x_test_new = pca.transform(x_test[:,:-1])
        x_test = np.column_stack([x_test_new, x_test[:,-1]])

    X = X_train[:,:-1].tolist()
    Y = X_train[:,-1].tolist()
    X, X_min, X_max = scale_data(X, train=True)
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
    elif k == '2':
        return gaussian_kernel(x,z,sigma)
    elif k == '3':
        return sigmoid_kernel(x,z,a,theta)
    elif k == '1':
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



def train(X, Y, x_test, y_test, svm_type, kernel_type, degree, cost, tolerance, n_crossval):
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
	prob = svm_problem(Y, X)
	parameter = '-s ' + str(svm_type) + ' -t '+ str(kernel_type) + ' -d ' + str(degree) + ' -c ' + str(cost) + ' -e ' + str(tolerance) + ' -v ' + str(n_crossval)
	print(f'parameters: {parameter}')

	param = svm_parameter(parameter)
	m = libsvm.svm_train(prob, param)

	Y_true = y_test

	X_new = []
	for i in range(x_test.shape[0]):
	    d = {}
	    for j in range(x_test.shape[1]):
	        d[j+1] = x_test[i,j]
	    X_new.append(d)

	Y_pred = []
	for i in range(x_test.shape[0]):
	    x0, _ = gen_svm_nodearray(X_new[i])
	    label = libsvm.svm_predict(m, x0)
	    Y_pred.append(label)

	ACC, MSE, SCC = evaluations(Y_true, Y_pred)

	return Y_pred, ACC, MSE, SCC


def predict(m, x_test, y_test):
	Y_true = y_test

	X_new = []
	for i in range(x_test.shape[0]):
	    d = {}
	    for j in range(x_test.shape[1]):
	        d[j+1] = x_test[i,j]
	    X_new.append(d)

	Y_pred = []
	for i in range(x_test.shape[0]):
	    x0, _ = gen_svm_nodearray(X_new[i])
	    label = libsvm.svm_predict(m, x0)
	    Y_pred.append(label) 

	ACC, MSE, SCC = evaluations(Y_true, Y_pred)

	return Y_pred, ACC, MSE, SCC 



def main(mode, kernel_type):
	X, Y, x, y = get_data(mode)
	x_test = x
	y_test = y
	Y_true = y_test

	degree = 3
	cost = 1
	tol = 0.001
	n_crossval = 5

	# classification
	if mode!=3:
		svm_type = 0
		
		Y_pred, ACC, MSE, SCC = train(X, Y, x_test, y_test, svm_type=svm_type ,kernel_type=kernel_type, degree=degree, cost=cost, tolerance=tol, n_crossval=n_crossval)
		
		acc = performanceAnalyser.calcAccuracyTotal(Y_pred,Y_true)
		precision, recall, f1score = performanceAnalyser.goodness(Y_true, Y_pred)
		confMat = performanceAnalyser.getConfusionMatrix(Y_true,Y_pred)
		print(f'\n\nAccuracy: {acc}\nPrecision: {precision}\n Recall: {recall}\nF1score: {f1score}\nConfusion Matrix: {confMat}\n')
	
	else: #regression
		svm_type = 4

		Y_pred, ACC, MSE, SCC = train(X, Y, x_test, y_test, svm_type=svm_type ,kernel_type=kernel_type, degree=degree, cost=cost, tolerance=tol, n_crossval=n_crossval)
		rmse = performanceAnalyser.calcRootMeanSquareRegression(np.asarray(Y_pred),np.asarray(Y_true))
		exp_var = explained_variance_score(Y_true, Y_pred)
		mse, r2 = performanceAnalyser.R2(np.asarray(Y_pred),np.asarray(Y_true))
		print(f'\n\nMSE: {mse}\nRMSE: {rmse}\nR2: {r2}\nExplained Variance: {exp_var}\n')


if __name__ == '__main__':
	data = {0:'Medical', 1:'F-MNIST', 2:'Railway', 3:'River'}
	k = {0: 'linear', 1:'gaussian', 2:'polynomial', 3:'sigmoid'}
	mode = int(sys.argv[1])
	kernel_type = int(sys.argv[2])
	print (f'--> {data[mode]} dataset')
	print (f'--> {k[kernel_type]} kernel\n')
	main(mode, kernel_type)




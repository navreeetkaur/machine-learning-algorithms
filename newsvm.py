import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import scipy 
from inputReader import InputReader 
import performanceAnalyser
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl


# prepare data for multiclass SVM - one vs one
def get_data(mode):
    if mode==0:
        inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
    elif mode==1:
        inputDataClass = InputReader(['fashion-mnist_train.csv', 'fashion-mnist_test.csv'],1)
    elif mode==2:
        inputDataClass = InputReader('railwayBookingList.csv',2)
    else:
    	inputDataClass = None
    	print('INVALID MODE')
    	exit()

    X_train = inputDataClass.Train
    X_train = inputDataClass.Train
    x_test = inputDataClass.Test
    if mode==1:
        pca = PCA(n_components=28)
        X_new = pca.fit_transform(X_train[:,:-1])
        X_train = np.column_stack([X_new, X_train[:,-1]])
        x_test_new = pca.transform(x_test[:,:-1])
        x_test = np.column_stack([x_test_new, x_test[:,-1]])


    Y = list(set(np.asarray(X_train[:,-1].tolist())))
    tot_c = len(Y)
    n = X_test.shape[0]

	p = itertools.permutations(Y, 2)
	classes = []
	for i in p:
		classes.append(i)
	L = []
	for c in classes:
		c0 = c[0]
		c1 = c[1]
		idx0 = np.where(X_train[:,-1]==c0)[0].tolist()
		idx1 = np.where(X_train[:,-1]==c1)[0].tolist()
		X = X_train[idx0+idx1,:-1]
		Y = X_train[idx0+idx1,-1].tolist()
		for j in range(len(Y)):
			if Y[i]==c0:
				Y[i]=-1
			else:
				Y[i]=1
		idx0 = np.where(X_test[:,-1]==c0)[0].tolist()
		idx1 = np.where(X_test[:,-1]==c1)[0].tolist()
		x = X_test[idx0+idx1,:-1]
		y = X_test[idx0+idx1,-1].tolist()
		for j in range(len(y)):
			if y[i]==c0:
				y[i]=-1
			else:
				y[i]=1
		L.append([c, idx0+idx1, X,Y,x,y])

	return L, n ,tot_c


def scale_data(X, train, X_min=None, X_max=None):
    X = np.asarray(X)
    if train:
        X_min = np.amin(X, axis=0)
        X_max = np.amax(X, axis=0)
    X = 2*np.divide(np.subtract(X,X_min), np.subtract(X_max, X_min)) - 1
    return X, X_min, X_max


def kernel(k, x):
	Kernel = np.zeros((x.shape[0],x.shape[0]))
	for i in range(x.shape[0]):
		for j in range(x.shape[0]):
			Kernel[i][j] = kernel_indv(k, x[i],x[j])
	return Kernel


def kernel_indv(k,x,z):
	if k == 0:
		return linear_kernel(x,z)
	elif k == 1:
		return gaussian_kernel(x,z)
	elif k == 2:
		return sigmoid_kernel(x,z)
	elif k == 3:
		return polynomial_kernel(x,z)


# types of kernels
def linear_kernel(x, y, b=1):
    return np.dot(x,y) + b


def gaussian_kernel(x, y, sigma=1):
    return np.exp((-np.linalg.norm(x-y)**2)/(2*(sigma**2)))


def sigmoid_kernel(x, y, a=1, theta=1):
    return numpy.tanh(a*np.matmul(x, y.transpose())+ theta)


def polynomial_kernel(x, y, b=1, degree=2):
    return (b + np.dot(x, y))**degree



def define_params(X, Y, k_mode, C=None):
	K = kernel(k_mode, X)
	n = len(Y)

	Y = np.asarray(Y)

	P = matrix(np.outer(Y,Y) * K)
	q = matrix((-1)*np.ones((n,1)))
	A = matrix(np.reshape(Y, (1,n)))
	b = matrix(0.0)

	if C is None:
            G = matrix(np.diag(np.ones(n) * -1))
            h = matrix(np.zeros(n))
        else:
            tmp1 = np.diag(np.ones(n) * -1)
            tmp2 = np.identity(n)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n)
            tmp2 = np.ones(n) * C
            h = matrix(np.hstack((tmp1, tmp2)))

	return K, P, q, G, h, A, b



def solve_opt(P, q, G, h, A, b):
	sol = solvers.qp(P, q, G, h, A, b)
	return np.ravel(sol['x'])


def get_svs(X, Y, K, k_mode, mus):
	# get mus greater than zero
	sv = mus > 1e-5
    idx = np.arange(len(mus))[sv]
    mus = mus[sv]
    # get support vectors
    sup_vecs = X[sv]
    sup_vecs_y = Y[sv]
    print(f'{len(mus)} support vectors out of {X.shape[0]} points')
    # intercept
    b = 0
    for i in range(len(mus)):
        b += sup_vecs[i]
        b -= np.sum(mus * sup_vecs_y * K[idx[i],sv])
    b /= len(mus)

    return b, mus, sup_vecs_y, sup_vecs


def predict(b, X_test, k_mode, mus, sup_vecs_x, sup_vecs_y):
	Y_pred = []
	for i in range(x.shape[0]):
		Y_pred.append(np.sign(f_x(mus, sup_vecs_x, sup_vecs_y, X_test[i], k_mode, b)))
	return Y_pred


# predict for a given test point
def f_x(mus, sup_vecs_x, sup_vecs_y, x_test, k_mode, b):
	func = 0.0
	for i in range(len(mus)):
		func += mus[i] * sup_vecs_y[i] * kernel_indv(k_mode, sup_vecs_x[i], x_test)
	return func + b


if __name__ == '__main__':
	mode = 0
	k_mode = 0
	predicted_y = []
	L, n, tot_c = get_data(mode)
	predictions = np.zeros((n,tot_c))
	for l in L:
		classes, idx, X, Y, x, y = l
		K, P, q, G, h, A, b = define_params(X, Y, k_mode, 1)
		mus = solve_opt(P, q, G, h, A, b)
		b, mus, sup_vecs_y, sup_vecs = get_svs(X, Y, K, k_mode, mus)
		Y_pred = predict(b, X_test, k_mode, mus, sup_vecs_x, sup_vecs_y)
		actual_Y_pred = []
		for i in range(len(Y_pred)):
			if Y_pred[i]==-1:
				z = classes[0]
			else:
				z = classes[1]
			actual_Y_pred.append(z)
			predictions[idx[i], z] += 1

	for i in range(predictions.shape[0]):
		predicted_y.append(np.where(max(predicted_y[i]))[0][0]+1)
		






	
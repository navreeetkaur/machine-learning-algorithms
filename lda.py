import numpy as np 
import pandas as pd
from scipy.linalg import eigh
from inputReader import InputReader 
import performanceAnalyser

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import scipy.misc



# distance measure
def distance(mode, data1, data2):
	# euclidean
	if mode == 0:
		n = len(data1)
		dis = 0
		for i in range(0,n):
			dis+=pow(data1[i]-data2[i],2)
		dis = pow(dis,0.5)
		return dis

	# manhattan
	if mode == 1:
		n = len(data1)
		dis = 0
		for i in range(0,n):
			dis+=math.fabs(data1[i]-data2[i])
		return dis

	# chebyshev
	if mode == 2:
		n = len(data1)
		dis = 0
		for i in range(0,n):
			x = math.fabs(data1[i]-data2[i])
			if x > dis:
				dis=x
		return dis

	# mahalanobis
	if mode == 3:
		dis = np.matmul(np.matmul((data1 - data2).transpose(),self.covarinv),data1-data2)
		return dis**0.5

	# cosine
	if mode==4:	
		num = np.dot(data1,data2)
		denom = (1.0*np.linalg.norm(data1))*np.linalg.norm(data2)
		dis = 1 - (num/(denom*1.0))
		return dis

def LDA(X, x_test, dist_metric = 0):
    #sorting according to Y
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

    # classes
    classes = list(sliced_matrix.keys())
    # number of classes
    n_classes = len(classes)
    #dimensions
    d = sliced_matrix[classes[0]].shape[1]

    S_within = np.zeros((d, d))
    S_between = np.zeros((d, d))
    # total_mean = np.zeros((d,))
    means = {}

    total_mean = np.sum(X[:,:-1])/(1.0*X.shape[0])
    # within class variance
    for i in range(len(classes)):
        c = classes[i]
        c_samples = sliced_matrix[c]
        #number of samples for this class
        n_curr = c_samples.shape[0]
        c_mean = c_samples.sum(axis=0)/(1.0*n_curr)
        means[c] = c_mean
        new_c_samples = c_samples - c_mean
        c_covar = np.matmul(new_c_samples.transpose(), new_c_samples)
        # total_mean = total_mean + n_curr*c_mean 
        # within class variance
        S_within = S_within + c_covar
        # between class variance
        new_c_samples = c_mean - total_mean
        c_covar = n_curr*(np.matmul(new_c_samples.transpose(), new_c_samples))
        S_between = S_between + c_covar

        i+=1

    # total_mean = total_mean/n_classes

    # # between class  variance
    # for c in classes:
    #     c_samples = sliced_matrix[c]
    #     #number of samples for this class
    #     n_curr = c_samples.shape[0]
    #     # c_mean = c_samples.sum(axis=0)/(1.0*n_curr)
    #     c_mean = means[c]
    #     new_c_samples = c_mean - total_mean
    #     c_covar = n_curr*(np.matmul(new_c_samples.transpose(), new_c_samples))
    #     S_between = S_between + c_covar


    # final_mat = np.matmul(np.linalg.inverse(S_within), S_between)
    # eigvals, eigvecs = np.linalg.eig(final_mat)
    eigvals, eigvecs = eigh(S_between, S_within, eigvals_only=False)
    # sort eigenvalues in decreasing order and obtain corresponding eigenvectors
    eig_sort = eigvals.argsort()
    eigvals = eigvals[eig_sort[::-1]]
    eigvecs = eigvecs[eig_sort[::-1]]
    W = eigvecs[:n_classes-1]
#     W = eigvecs


    Y_pred = []
    # classification given a new instance 'z'
    for i in range(x_test.shape[0]):
        z = x_test[i][:-1]
        # argmin 
        dist = float("inf")
        for c in classes:
            a, b = np.matmul(W, z.transpose()), np.matmul(W, means[c].transpose())
            curr_dist = distance(dist_metric, a, b)
            if curr_dist<dist:
                dist = curr_dist
                curr_class = c
        Y_pred.append(curr_class)

    # analyse performance
    Y_true = x_test[:,-1]
    acc = performanceAnalyser.calcAccuracyTotal(Y_pred,Y_true)
    precision, recall, f1score = performanceAnalyser.goodness(Y_true, Y_pred)
    confMat = performanceAnalyser.getConfusionMatrix(Y_true,Y_pred)

    print(f'Accuracy:{acc}\n Precision:{precision}\n Recall:{recall}\n F1score:{f1score}\n Confusion Matrix:{confMat}\n')

    return Y_pred, acc, precision, recall, f1score, confMat


def calcPhiX(X):
    basis = []
    n=3
    for i in range(n):
        basis.append(X**i)
    phiX = np.hstack(basis)
    return phiX


if __name__ == '__main__':
	# get data
	# inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
	inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
	X = inputDataClass.Train
	x_test = inputDataClass.Test

	####################### TSNE #####################

	# tsne = TSNE(n_components=2, random_state=0)
	# np.set_printoptions(suppress=True)

	# X_new = tsne.fit_transform(X)
	# X_new = np.column_stack((X_new,X[:,-1]))

	# x_test_new = tsne.fit_transform(x_test)
	# x_test_new = np.column_stack((x_test_new, x_test[:,-1]))

	# x_coords = X_new[:, 0]
	# y_coords = X_new[:, 1]
	# x_test_coords = X_new[:, 0]
	# y_test_coords = X_new[:, 1]
	# fig = plt.figure(figsize=(8, 6))
	# plt.scatter(x_coords, y_coords, c=X[:,-1])
	    
	# plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
	# plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
	# plt.title("TSNE plot of medical data")
	# plt.show()

	####################### Kernel Transformation #####################
	X_new = calcPhiX(X[:,:-1])
	X_new = np.column_stack((X_new, X[:,-1]))
	x_test_new = calcPhiX(x_test[:,:-1])
	x_test_new = np.column_stack((x_test_new,x_test[:,-1]))

	print("For ORIGINAL data. . .")
	Y_pred, acc, precision, recall, f1score, confMat = LDA(X, x_test)
	# print("\nFor t-SNE transformed data. . .")
	# Y_pred, acc, precision, recall, f1score, confMat = LDA(X_new, x_test_new)
	print("For transformed kernel .. . ")
	Y_pred, acc, precision, recall, f1score, confMat = LDA(X_new, x_test_new)
	




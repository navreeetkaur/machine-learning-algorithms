import numpy as np 
import pandas as pd
from scipy.linalg import eigh
from inputReader import InputReader 
import performanceAnalyser

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import scipy.misc

from Distributions import gaussian_multivar

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

def LDA(X, x_test, mode = 0, max_dim=None, dist_metric = 0):
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
    if max_dim is None:
    	max_dim = n_classes-1

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


    # mat = np.dot(np.linalg.pinv(S_within), S_between)
	# eigvals, eigvecs = np.linalg.eig(mat)
    eigvals, eigvecs = eigh(S_between, S_within, eigvals_only=False)
    # sort eigenvalues in decreasing order and obtain corresponding eigenvectors
    eig_sort = eigvals.argsort()
    eigvals = eigvals[eig_sort[::-1]]
    eigvecs = eigvecs[eig_sort[::-1]]
    W = eigvecs[:max_dim]

	####################### thresholding #####################
	# for projection on 1-D, 2-class classification
	'''
	Calculate the classification threshold.
	Projects the means of the classes and takes their mean as the threshold.
	Also specifies whether values greater than the threshold fall into class 1 
	or class 2.
	'''
	# if n_classes==2 and mode == 0:
	# 	total = 0
	# 	for c in classes:
	# 	    total += np.matmul(W, means[c].transpose())

	# 	W0 = np.asarray(0.5 * total)
	# 	c0 = list(means.keys())[0]
	# 	c1 = list(means.keys())[1]
	# 	mu0 = np.asarray(np.matmul(W, means[c0].transpose()))
	# 	'''
	# 	calculate the scores in thresholding method.
	# 	Assigns predictions based on the calculated threshold.
	# 	'''
	# 	# project the inputs
	# 	proj = np.matmul(W, x_test[:,:-1].transpose()).transpose()
	# 	if all(np.greater_equal(mu0,W0)):
	# 	    Y_pred = [c0 if all(np.greater_equal(proj[i],W0)) else c1 for i in range(len(proj))]
	# 	else:
	# 	    Y_pred = [c0 if all(np.greater(W0, proj[i])) else c1 for i in range(len(proj))]


	####################### guassian modelling ################	
	'''
	Estimate gaussian models for each class.
	Estimates priors, means and covariances for each class.
	'''
	if mode==1:
	    priors = {}
		gaussian_means = {}
		gaussian_cov = {}

		for c in classes:
		    c_samples = sliced_matrix[c]
		    proj = np.matmul(W, c_samples.transpose()).transpose()
		    priors[c] = c_samples[:,:-1].shape[0] / float(x_test[:,:-1].shape[0])
		    gaussian_means[c] = np.mean(proj, axis = 0)
		    gaussian_cov[c] = np.cov(proj, rowvar=False)

		# Calculate error rates based on gaussian modeling.
		# project the test data
		proj = np.matmul(W, x_test[:,:-1].transpose()).transpose()
		# calculate the likelihoods for each class based on the gaussian models
		likelihoods = np.array([[priors[c] * gaussian_multivar(np.asarray([x[ind] for ind in 
		                                                    range(len(x))]), gaussian_means[c], 
		                                                    gaussian_cov[c]) for c in classes] for x in proj])
		# assign prediction labels based on the highest probability
		Y_pred = np.argmax(likelihoods, axis = 1)


	###################classification given a new instance 'z' ###################
	if mode==2:
	    Y_pred = []
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
    for i in range(1, n+1):
        basis.append(X**i)
    phiX = np.hstack(basis)
    return phiX


if __name__ == '__main__':
	# get data
	inputDataClass = InputReader(['fashion-mnist_train.csv', 'fashion-mnist_test.csv'],1)
	# inputDataClass = InputReader(['Medical_data.csv', 'test_medical.csv'],0)
	# inputDataClass = InputReader('railwayBookingList.csv',2)

	X = inputDataClass.Train
	x_test = inputDataClass.Test

	# PCA for F-MNIST
	# pca = PCA(n_components=80)
	# X_new = pca.fit_transform(X[:,:-1])
	# X = np.column_stack([X_new, X[:,-1]])
	# x_test_new = pca.transform(x_test[:,:-1])
	# x_test = np.column_stack([x_test_new, x_test[:,-1]])

	####################### TSNE #####################
	######## transformation ##########

	# tsne = TSNE(n_components=2, random_state=0)
	# np.set_printoptions(suppress=True)

	# X_new = tsne.fit_transform(X)
	# X_new = np.column_stack((X_new,X[:,-1]))

	# x_test_new = tsne.fit_transform(x_test)
	# x_test_new = np.column_stack((x_test_new, x_test[:,-1]))

	######## plot ##########
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

	print("For ORIGINAL data. . .")
	Y_pred, acc, precision, recall, f1score, confMat = LDA(X, x_test, mode=1)
	# print("\nFor t-SNE transformed data. . .")
	# Y_pred, acc, precision, recall, f1score, confMat = LDA(X_new, x_test_new)
	# print("For transformed kernel .. . ")
	# Y_pred, acc, precision, recall, f1score, confMat = LDA(X_new, x_test_new)
	print("SKLEARN. . .")
	model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
	model.fit(X[:,:-1], X[:,-1])
	Y_pred = model.predict(x_test[:,:-1])
	# analyse performance
	Y_true = xx[:,-1]
	acc = performanceAnalyser.calcAccuracyTotal(Y_pred,Y_true)
	precision, recall, f1score = performanceAnalyser.goodness(Y_true, Y_pred)
	confMat = performanceAnalyser.getConfusionMatrix(Y_true,Y_pred)
	print(f'Accuracy:{acc}\n Precision:{precision}\n Recall:{recall}\n F1score:{f1score}\n Confusion Matrix:{confMat}\n')


	




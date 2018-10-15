import numpy as np
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import inputReader
import Bayes
import performanceAnalyser
import Preprocessing
import kmeans
import KNN
import Visualization
import ROC
import linearModels

dists = {-1: "Ignore",0:"Gaussian", 1:"Multinomail"}

def performPCA(inputDataClass,reduced_columns):
	############################################## PCA Visualisation #############################################
	# #variance v/s n_components : Fashion MNIST
	# start = 10
	# stop = 500
	# step = 15
	# Visualization.var_vs_comp(inputDataClass.Train[:,:-1], start, stop, step)
	########################################################### PCA #############################################

	##### Our PCA ####
	pca = Preprocessing.PCA(inputDataClass.Train[:,:-1], k = reduced_columns, whiten = False)					##### Hyperparameter ####
	reduced_train = pca.reduce(inputDataClass.Train[:,:-1], True)
	inputDataClass.Train =  np.hstack((reduced_train,inputDataClass.Train[:,-1].reshape(-1,1)))
	print("train_data reduced.")
	print("Train data reduced to columns = "+str(reduced_train.shape[1]))
	reduced_test = pca.reduce(inputDataClass.Test[:,:-1], False)
	inputDataClass.Test =  np.hstack((reduced_test,inputDataClass.Test[:,-1].reshape(-1,1)))
	print("test_data reduced. ")
	print("Test data reduced to columns = "+str(reduced_test.shape[1]))

	### SKlearn PCA #####
	# pca = PCA(n_components=80,whiten=False)
	# pca.fit(inputDataClass.Train[:,:-1])
	# reduced_train = pca.transform(inputDataClass.Train[:,:-1])
	# inputDataClass.Train =  np.hstack((reduced_train,inputDataClass.Train[:,-1].reshape(-1,1)))
	# reduced_test = pca.transform(inputDataClass.Test[:,:-1])
	# inputDataClass.Test =  np.hstack((reduced_test,inputDataClass.Test[:,-1].reshape(-1,1)))


def normalizeData(inputDataClass):
	######################################## Normalising Data ####################################
	normalizer = Preprocessing.Normalise()
	inputDataClass.Train = np.hstack((normalizer.scale(inputDataClass.Train[:,:-1],train=True),inputDataClass.Train[:,-1].reshape(-1,1)))
	inputDataClass.Test = np.hstack((normalizer.scale(inputDataClass.Test[:,:-1],train=False),inputDataClass.Test[:,-1].reshape(-1,1)))

def performVisualizations(inputDataClass):
	########################################### Visualizations ###################################################
	# Visualization.visualizeDataCCD(np.vstack((inputDataClass.Train,inputDataClass.Test)))

	correlation_dict = performanceAnalyser.getCorrelationMatrix(inputDataClass.Train)
	Visualization.visualizeCorrelation(correlation_dict)

	# Visualization.visualizeDataPoints(inputDataClass.Train)
	# Visualization.comp_vs_var_accuracy()
	# pass

def performBayes(inputDataClass, drawPrecisionRecall = False, drawConfusion = False):
	"""################################# Bayes Classifier #############################################"""

	##Sklearn
	# print("\nSklearn Naive Bayes")
	# clf = GaussianNB()
	# clf.fit(inputDataClass.Train[:,:-1], inputDataClass.Train[:,-1])

	# Ypred = clf.predict(inputDataClass.Train[:,:-1])
	# Ytrue = inputDataClass.Train[:,-1]
	# print("Training Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))

	# Ypred = clf.predict(inputDataClass.Test[:,:-1])
	# Ytrue = inputDataClass.Test[:,-1]
	# print("Testing Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))


	print("\nMy Naive Bayes")
	# bayesClassifier = Bayes.Bayes(isNaive = False, distribution =[0 for i in range(inputDataClass.Train.shape[1]-1)])
	bayesClassifier = Bayes.Bayes(isNaive = True, distribution =[0,0,1,1,0])
	bayesClassifier.train(inputDataClass.Train)
	print("Training of model done.")

	Ypred = bayesClassifier.fit(inputDataClass.Train)
	Ytrue = inputDataClass.Train[:,-1]
	print("Training Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))

	Ypred = bayesClassifier.fit(inputDataClass.Test)
	Ytrue = inputDataClass.Test[:,-1]
	print("Testing Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))

	print("Prediction done.")

	if drawConfusion:
		confusion = performanceAnalyser.getConfusionMatrix(Ytrue,Ypred)
		Visualization.visualizeConfusion(confusion)

	if drawPrecisionRecall:		
		############################ precision-recall curve #############################
		threshold = np.arange(0.9,0.1,-0.1)
		probas = bayesClassifier.get_probas()
		for dic in probas:
			sums=0.0
			for item in dic:
				sums+=dic[item]
			for item in dic:
				dic[item] = dic[item]/sums
		roc = ROC.Roc(Ytrue,probas,threshold,'')
		roc.Roc_gen()

		precision, recall, _ = precision_recall_curve(Ytrue, probas)

		plt.step(recall, precision, color='b', alpha=0.2, where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision Recall Curve')

	return Ytrue,Ypred

def performKMeans(inputDataClass,k,mode,num_runs,visualize=False):
	covar = -1
	if mode == 3:
		covar = performanceAnalyser.getFullCovariance(inputDataClass.Train[:,:-1])
	labels, means, rms, Ypred = kmeans.kfit(inputDataClass.Train[:,:-1],k,inputDataClass.Train[:,-1],inputDataClass.Test[:,:-1],num_runs = num_runs, mode = mode,covar=covar)
	print("rms = "+str(rms))
	print("Kmeans done")

	Ytrue = inputDataClass.Test[:,-1]
	print("Testing Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))

	if visualize:
		Visualization.visualizeKMeans(inputDataClass.Train[:,:-1],labels,k)
		print("Kmeans visualized")

	return Ytrue,Ypred

def performKNN(inputDataClass, nearestNeighbours,mode,label_with_distance=False):
	covar=-1
	if mode == 3:
		covar = performanceAnalyser.getFullCovariance(inputDataClass.Train[:,:-1])
	knn = KNN.KNN(nearestNeighbours,inputDataClass.Train[:,:-1],inputDataClass.Test[:,:-1],inputDataClass.Train[:,-1],label_with_distance=label_with_distance, mode=mode, covar=covar)
	knn.allocate()
	Ypred = knn.labels
	Ytrue = inputDataClass.Test[:,-1]
	print("Testing Accuracy = "+str(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)))
	return Ytrue,Ypred

def performLinearModels(inputDataClass, phiMode, maxDegree, isRegularized, lambd, isRegress = False):
	linear_model = linearModels.LinearModels(phiMode,maxDegree,isRegularized,lambd)
	linear_model.train(inputDataClass.Train)
	Ypred = linear_model.test(inputDataClass.Test[:,:-1], isRegress)
	Ytrue = inputDataClass.Test[:,-1]
	if isRegress:
		rms = performanceAnalyser.calcRootMeanSquareRegression(Ypred,Ytrue)
		print("Linear model rms "+str(rms))
	if not isRegress:
		acc = performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue)
		print("Linear model Accuracy "+str(acc))
	return Ytrue,Ypred

def performMultiClassLinear(inputDataClass,phiMode,maxDegree,learnRate):
	multi_class_linear_model = linearModels.MultiClassLinear(phiMode,maxDegree,learnRate)
	multi_class_linear_model.train(inputDataClass.Train)
	Ypred = multi_class_linear_model.test(inputDataClass.Test[:,:-1])

if __name__ == '__main__': 
	if len(sys.argv) < 2:
		print("Invalid Format. Provide input file names")
		exit()
	inputDataFile = sys.argv[1]
	
	mode = -1		# 0 for Medical; 1 for Fashion; 2 for Railway

	mod_dict = {0:'Medical_data', 1:'fashion-mnist', 2:'railway_Booking', 3:'River Data'}

	if inputDataFile == 'Medical_data.csv':
		mode = 0
		x= []
		x.append(inputDataFile)
		if len(sys.argv) != 3:
			print('Enter both train and test files')
			exit()
		x.append(sys.argv[2])
		inputDataFile = x
	elif inputDataFile == 'fashion-mnist_train.csv':
		mode = 1
		x= []
		x.append(inputDataFile)
		if len(sys.argv) != 3:
			print('Enter both train and test files')
			exit()
		x.append(sys.argv[2])
		inputDataFile = x
	elif inputDataFile == 'railwayBookingList.csv':
		mode = 2
	elif inputDataFile == 'river_data.csv':
		mode =3

	if mode==-1:
		print("Unknown Dataset. Enter valid dataset.")
		exit()


	train_test_ratio = 0.8
	inputDataClass = inputReader.InputReader(inputDataFile,mode,train_test_ratio = train_test_ratio)

	if mode == 1:
		"""################################# PCA #############################################"""
		reduced_columns = 80
		performPCA(inputDataClass = inputDataClass, reduced_columns = reduced_columns)	
	
	"""################################# Normalisation #############################################"""
	normalizeData(inputDataClass = inputDataClass)

	"""################################# Visualization #############################################"""
	# performVisualizations(inputDataClass = inputDataClass)

	"""################################# Bayes #############################################"""
	# Ytrue,Ypred = performBayes(inputDataClass = inputDataClass, drawPrecisionRecall = False, drawConfusion = False)

	"""################################# Linear Models #############################################"""
	# PHIMODE => 0 : Projection ;;; 1 : 1,x,x2 (Add maxDegree)
	# Ytrue,Ypred = performLinearModels(inputDataClass = inputDataClass, phiMode=0, maxDegree=1, isRegularized = False, lambd = 1, isRegress = False)
	performMultiClassLinear(inputDataClass = inputDataClass, phiMode = 0, maxDegree = 1, learnRate = 0.1, isRegularized = True, lambd = 0.1)
	
	"""################################# KMEANS #############################################"""
	# k = 3					### Hyperparameter ###
	# mode = 0			# mode = {0 : Euclidean, 1: Manhattan, 2 : Chebyshev, 3: Mahalnobis}
	# num_runs= 100
	# Ytrue,Ypred = performKMeans(inputDataClass,k,mode,num_runs,visualize=False)

	"""################################# KNN #############################################"""

	# nearestNeighbours = 15	### Hyperparameter ###	
	# mode = 0		# mode = {0 : Euclidean, 1: Manhattan, 2 : Chebyshev}
	# performKNN(inputDataClass, nearestNeighbours,mode,label_with_distance=False)	

	"""###############################PRECISION-RECALL-F1##########################################"""
	# print(Ytrue)
	# print(Ypred)
	# precision,recall, f1score = performanceAnalyser.goodness(Ytrue,Ypred)
	# print("\nPrecision")
	# print(precision)
	# print("Recall")
	# print(recall)
	# print("F1 Score")
	# print(f1score)
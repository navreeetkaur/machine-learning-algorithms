import numpy as np
import sys

import Bayes,performanceAnalyser

class Classifier:
	train_test_ratio = 0.8
	def __init__(self,inputDataFileList,mode):
		self.mode = mode
		if mode == 0:
			self.Train, self.Test, self.Label = self.collectInputMedical(inputDataFileList)
		elif mode == 1:
			self.Train, self.Test, self.Label = self.collectInputFashion(inputDataFileList)
		else:
			self.Train, self.Test, self.Label = self.collectInputRailway(inputDataFileList)

	def collectInputMedical(self,inputDataFileList):
		with open(inputDataFile,'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			labels = ['TEST1','TEST2','TEST3','Health']
			num_labels = len(labels)
			num_records= len(lines)
			num_train = int(Classifier.train_test_ratio*num_records)
			num_test = num_records - num_train
			test_array = np.zeros((num_test,num_labels),dtype= np.float64)
			train_array = np.zeros((num_train,num_labels),dtype=np.float64)
			test_indices = np.sort(np.random.choice(num_records-1,num_test,replace=False))
			# print(test_indices)
			i=0
			for index in test_indices:
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				for j in range(num_labels-1):
					# print(j,num_labels)
					# print(record[j+1])
					test_array[i][j] = float(record[j+1])
				y_label = record[0]
				if y_label == 'HEALTHY':
					test_array[i][3] = 1
				elif y_label == 'MEDICATION':
					test_array[i][3] = 2
				elif y_label == 'SURGERY':
					test_array[i][3] = 3
				if test_array[i][3] == 0:
					print("Invalid treatment type detected at line "+int(index+2))
					exit()
				i+=1

			i=0
			for index in range(num_records):
				if index in test_indices:
					continue
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				for j in range(num_labels-1):
					train_array[i][j] = float(record[j+1])
				y_label = record[0]
				if y_label == 'HEALTHY':
					train_array[i][3] = 1
				elif y_label == 'MEDICATION':
					train_array[i][3] = 2
				elif y_label == 'SURGERY':
					train_array[i][3] = 3
				if train_array[i][3] == 0:
					print("Invalid treatment type detected at line "+int(index+2))
					exit()
				i+=1

		# print(train_array)
		# print(test_array)
		# print(labels)
		return train_array,test_array,labels


	def collectInputFashion(self,inputDataFileList):
		print(inputDataFileList)
		# Loading training data
		labels = ['pixel'+str(i+1) for i in range(784) ]
		labels.append('Class')
		num_labels = len(labels)
		with open(inputDataFileList[0],'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			num_records= len(lines)
			train_array = np.zeros((num_records,num_labels),dtype=np.int)
			for i in range(num_records):
				record = lines[i]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				for j in range(num_labels-1):
					train_array[i][j] = int(record[j+1])
				train_array[i][num_labels-1] = int(record[0])

		with open(inputDataFileList[1],'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			num_records= len(lines)
			test_array = np.zeros((num_records,num_labels),dtype=np.int)
			for i in range(num_records):
				record = lines[i]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				for j in range(num_labels-1):
					test_array[i][j] = int(record[j+1])
				test_array[i][num_labels-1] = int(record[0])

		# print(train_array)
		# print(test_array)
		# print(labels)
		return train_array,test_array,labels


	def collectInputRailway(self,inputDataFileList):
		with open(inputDataFile,'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			labels = ['caseID','budget','memberCount','preferredClass','sex','age','ifBoarded']
			num_labels = len(labels)
			num_records= len(lines)
			num_train = int(Classifier.train_test_ratio*num_records)
			num_test = num_records - num_train
			test_array = np.zeros((num_test,num_labels),dtype= np.int64)
			train_array = np.zeros((num_train,num_labels),dtype=np.int64)
			test_indices = np.sort(np.random.choice(num_records-1,num_test,replace=False))

			i=0
			for index in test_indices:
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				
				#Handling caseID
				test_array[i][0] = int(record[0])
				
				#Handling budget
				test_array[i][1] = int(record[2])
				
				#Handling memberCount
				test_array[i][2] = int(record[3])
				
				#Handling preferredClass
				if record[4] == 'FIRST_AC':
					test_array[i][3] = 1
				elif record[4] == 'SECOND_AC':
					test_array[i][3] = 2
				elif record[4] == 'THIRD_AC':
					test_array[i][3] = 3
				elif record[4] == 'NO_PREF':
					test_array[i][3] = 4
				else:
					print(record[4])
					print('Unknown preferredClass detected')
					exit()

				# Handling sex
				if record[5] == 'male':
					test_array[i][4] = 1
				elif record[5] == 'female':
					test_array[i][4] = 2
				elif record[5] == '':
					test_array[i][4] = 3
				else:
					print(record[5])
					print('Unknown sex detected')
					exit()

				# Handling age
				test_array[i][5] = int(record[6])

				#Handling ifBoarded
				test_array[i][6] = int(record[1])

				i+=1

			i=0
			for index in range(num_records):
				if index in test_indices:
					continue
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				#Handling caseID
				train_array[i][0] = int(record[0])
				
				#Handling budget
				train_array[i][1] = int(record[2])
				
				#Handling memberCount
				train_array[i][2] = int(record[3])
				
				#Handling preferredClass
				if record[4] == 'FIRST_AC':
					train_array[i][3] = 1
				elif record[4] == 'SECOND_AC':
					train_array[i][3] = 2
				elif record[4] == 'THIRD_AC':
					train_array[i][3] = 3
				elif record[4] == 'NO_PREF':
					train_array[i][3] = 4
				else:
					print(record[4])
					print('Unknown preferredClass detected')
					exit()

				# Handling sex
				if record[5] == 'male':
					train_array[i][4] = 1
				elif record[5] == 'female':
					train_array[i][4] = 2
				elif record[5] == '':
					train_array[i][4] = 3
				else:
					print(record[5])
					print('Unknown sex detected')
					exit()

				# Handling age
				train_array[i][5] = int(record[6])

				#Handling ifBoarded
				train_array[i][6] = int(record[1])

				i+=1

			# print(train_array)
			# print(test_array)
			# print(labels)
			return train_array,test_array,labels	


if __name__ == '__main__': 
	if len(sys.argv) < 2:
		print("Invalid Format. Provide input file names")
		exit()
	inputDataFile = sys.argv[1]
	""" Fashion MNIST has separate files for training and test """
	mode = -1		# 0 for Medical; 1 for Fashion; 2 for Railway

	mod_dict = {0:'Medical_data', 1:'fashion-mnist', 2:'railway_Booking'}

	if inputDataFile == 'Medical_data.csv':
		mode = 0
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

	if mode==-1:
		print("Unknown Dataset. Enter valid dataset.")
		exit()

	inputDataClass = Classifier(inputDataFile,mode)
	performanceAnalyser = performanceAnalyser.PerformanceCheck()
	bayesClassifier = Bayes.Bayes(isNaive = False, distribution =[0,0,0,0,0,0])
	bayesClassifier.train(inputDataClass.Train)
	Ypred = bayesClassifier.fit(inputDataClass.Test)
	
	Ytrue = inputDataClass.Test[:,-1]
	# print(inputDataClass.Train)
	# print(inputDataClass.Test)

	print(Ytrue)
	print(Ypred)
	print(performanceAnalyser.calcAccuracyTotal(Ypred,Ytrue))





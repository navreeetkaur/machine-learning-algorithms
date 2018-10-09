import numpy as np

class InputReader:
	dataset_folder = 'Datasets'	

	def __init__(self,inputDataFileList,mode,train_test_ratio = 0.8):
		self.mode = mode
		self.train_test_ratio = train_test_ratio
		if mode == 0:
			self.Train, self.Test, self.Label = self.collectInputMedical(inputDataFileList)
		elif mode == 1:
			self.Train, self.Test, self.Label = self.collectInputFashion(inputDataFileList)
		elif mode == 2:
			self.Train, self.Test, self.Label = self.collectInputRailway(inputDataFileList)
		else:
			self.Train, self.Test, self.Label = self.collectInputRiver(inputDataFileList)

	def collectInputMedical(self,inputDataFileList):
		
		labels = ['TEST1','TEST2','TEST3','Health']
		num_labels = len(labels)
		arrays = []

		for k in range(len(inputDataFileList)):
			file = inputDataFileList[k]
			with open(str(self.dataset_folder+'/'+ file),'r') as inputFile:
				lines = inputFile.readlines()
				lines = lines[1:]
				num_records= len(lines)
				arrays.append(np.zeros((num_records,num_labels),dtype=np.float64))
				for i in range(num_records):
					record = lines[i]
					if record.strip() == '':
						continue
					record = record.strip().split(',')
					for j in range(num_labels-1):
						arrays[k][i][j] = float(record[j+1])
					y_label = record[0]
					if y_label == 'HEALTHY':
						arrays[k][i][3] = 0
					elif y_label == 'MEDICATION':
						arrays[k][i][3] = 1
					elif y_label == 'SURGERY':
						arrays[k][i][3] = 2
					if arrays[k][i][3] == -1:
						print("Invalid treatment type detected at line "+int(index+2)+" in file "+str(k+1))
						exit()

		return arrays[0],arrays[1],labels


	def collectInputFashion(self,inputDataFileList):

		labels = ['pixel'+str(i+1) for i in range(784) ]
		labels.append('Class')
		num_labels = len(labels)
		arrays = []

		for k in range(len(inputDataFileList)):
			file = inputDataFileList[k]
			with open(str(self.dataset_folder+'/'+ file),'r') as inputFile:
				lines = inputFile.readlines()
				lines = lines[1:]
				num_records= len(lines)
				arrays.append(np.zeros((num_records,num_labels),dtype=np.int))
				for i in range(num_records):
					record = lines[i]
					if record.strip() == '':
						continue
					record = record.strip().split(',')
					for j in range(num_labels-1):
						arrays[k][i][j] = int(record[j+1])
					arrays[k][i][num_labels-1] = int(record[0])

		return arrays[0],arrays[1],labels


	def collectInputRailway(self,inputDataFileList):
		with open(str(self.dataset_folder+'/'+ inputDataFileList),'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			labels = ['caseID','budget','memberCount','preferredClass','sex','age','ifBoarded']
			num_labels = len(labels)
			num_records= len(lines)
			num_train = int(self.train_test_ratio*num_records)
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
		return train_array,test_array,labels	

	def collectInputRiver(self,inputDataFileList):
		with open(str(self.dataset_folder+'/'+ inputDataFileList),'r') as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			labels = ['x','Levels']
			num_labels = len(labels)
			num_records= len(lines)
			num_train = int(self.train_test_ratio*num_records)
			num_test = num_records - num_train
			test_array = np.zeros((num_test,num_labels),dtype= np.float64)
			train_array = np.zeros((num_train,num_labels),dtype=np.float64)
			test_indices = np.sort(np.random.choice(num_records-1,num_test,replace=False))

			i=0
			for index in test_indices:
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')				
				#Handling x
				test_array[i][0] = float(record[0])
				#Handling O2 level
				test_array[i][1] = float(record[1])
				i+=1

			i=0
			for index in range(num_records):
				if index in test_indices:
					continue
				record = lines[index]
				if record.strip() == '':
					continue
				record = record.strip().split(',')
				#Handling x
				test_array[i][0] = float(record[0])
				#Handling O2 level
				test_array[i][1] = float(record[1])
				i+=1
		return train_array,test_array,labels
import numpy as np
import math

class PerformanceCheck:
	def __init__(self):
		pass

	def calcAccuracyTotal(self,Ypred,Ytrue):
		tot = len(Ypred)
		correct = 0
		for i in range(tot):
			if Ypred[i] == Ytrue[i]:
				correct+=1

		return correct*100.0/tot





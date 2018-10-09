import numpy as np 
from scipy.linalg import eigh


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

# classes
classes = sliced_matrix.keys()
# number of classes
n_classes = len(classes)
#dimensions
d = sliced_matrix[classes[0]].shape[1]

S_within = np.zeros((n_classes, n_classes))
S_between = np.zeros((n_classes, n_classes))
total_mean = np.zeros((d,))
means = {}

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
	total_mean = np.sum(total_mean, n_curr*c_mean)
	if i<(len(classes)-1):
		# only for c-1 classes
		S_within = S_within + c_covar
	i+=1

total_mean = total_mean/n_classes

# between class  variance
for c in classes:
	c_samples = sliced_matrix[c]
	#number of samples for this class
	n_curr = c_samples.shape[0]
	# c_mean = c_samples.sum(axis=0)/(1.0*n_curr)
	c_mean = means[c]
	new_c_samples = c_mean - total_mean
	c_covar = n_curr*(np.matmul(new_c_samples.transpose(), new_c_samples))
	S_between = S_between + c_covar


# final_mat = np.matmul(np.linalg.inverse(S_within), S_between)
# eigvals, eigvecs = np.linalg.eig(final_mat)
eigvals, eigvecs = eigh(S_between, S_within, eigvals_only=False)
# sort eigenvalues in decreasing order and obtain corresponding eigenvectors
eig_sort = eigvals.argsort()
eigvals = eigvals[eig_sort[::-1]]
eigvecs = eigvecs[eig_sort[::-1]]
W = eigvecs[:,:n_classes-1]


# classification given a new instance 'z'
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

# argmin 
dist = float("inf")
for c in classes:
	a, b = np.matmul(W.transpose(), z), np.matmul(W.transpose(), means[c])
	curr_dist = distance(0, a, b)
	if curr_dist<dist:
		dist = curr_dist
		curr_class = c









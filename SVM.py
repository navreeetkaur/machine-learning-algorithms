import numpy as np 
import scipy 

class SVM(object):
	def __init__(self, X, mus, errors, kernel='linear', C=1.0):
		self.X = X[:,:-1]
		self.Y = X[:,-1]
		self.k = kernel
		self.mus = []
		self.errors = errors
		self.C = C
		self.W = W
		self.b = b
		self.obj = []


	def linear_kernel(self, x, y, b=1):
		return np.matmul(x,y.transpose()) + b


	def gaussian_kernel(self, x, y, sigma=1):
		return np.exp(-(np.linalg.norm(x-y)/(2*(sigma**2))))


	def sigmoid_kernel(self, x, y, a=1, theta=1):
		return numpy.tanh(a*np.matmul(x, y.transpose())+ theta)


	def polynomial_kernel(self, x, y, b=1, degree=2):
		return (b + np.matmul(x, y.transpose()))**degree


	def kernel(self, x, z):
		if self.k=='linear':
			return self.linear_kernel(x,z)
		elif self.k == 'gaussian':
			return self.gaussian_kernel(x,z)
		elif self.k == 'sigmoid':
			return self.sigmoid_kernel(x,z)
		elif self.k = 'polynomial':
			return self.polynomial_kernel(x,z)


	def objection_fun(self):
		val = 0.
		for i in range(len(self.mus)):
			for j in range(len(self.mus)):
				val += self.mus[i]*self.mus[j]*self.Y[i]*self.Y[j]*kernel(self.X[i], self.X[j])
		return np.sum(mus) - (0.5 * val)


	def decision_fun(self, x_test):
		val = 0.
		for i in range(len(self.mus)):
			if self.mus[i]!=0:
				val += self.mus[i]*self.Y[i]*self.kernel(self.X[i], x_test) 
		return val + self.b


	def loss(self):
		loss = 0.0
		for i in range(len(self.Y)):
			loss += max(0, 1 - self.Y[i]*self.decision_fun(self.X[i]))
		return loss



def train(model):
    numChanged = 0
    examineAll = True

    while(numChanged > 0) or (examineAll):
        numChanged = 0
        if examineAll==1:
            # loop over all training examples
            for i in range(len(model.mus)):
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = model.objective_function()
                    model.obj.append(obj_result)
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.mus != 0) & (model.mus != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result==1:
                    obj_result = model.objective_function()
                    model.obj.append(obj_result)

        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
        
    return model



def examine_example(i2, model):
    
    y2 = model.Y[i2]
    mu2 = model.mus[i2]
    E2 = model.errors[i2]
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -tol and mu2 < model.C) or (r2 > tol and mu2 > 0)):
        
        if len(model.mus[(model.mus != 0) & (model.mus != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
            
        # Loop through non-zero and non-C alphas, starting at a random point
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
        
        # loop through all alphas, starting at a random point
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
    
    return 0, model


def take_step(i1, i2, model):
    
    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model
    
    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2
    
    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    eta = 2 * k12 - k11 - k22
    
    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H
            
    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L
        # objective function output with a2 = L
        Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X) 
        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
        if Lobj > (Hobj + eps):
            a2 = L
        elif Lobj < (Hobj - eps):
            a2 = H
        else:
            a2 = alph2
            
    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    
    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model
    
    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)
    
    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b
    
    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2
    
    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0
    
    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] + \
                            y1*(a1 - alph1)*model.kernel(model.X[i1], model.X[non_opt]) + \
                            y2*(a2 - alph2)*model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new
    
    # Update model threshold
    model.b = b_new
    
    return 1, model

		

		
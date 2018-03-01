import numpy as np


# Shuffles in unison 2 arrays
def shuffle_in_unison_2(a,b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

# Shuffles in unison 3 arrays
def shuffle_in_unison_3(a,b,c):
	assert len(a) == len(b) == len(c)
	p = np.random.permutation(len(a))
	return a[p], b[p], c[p]

def hessian(x):

	x_grad = np.gradient(x)
	hessian = np.empty((x.ndim,x.ndim) + x.shape, dtype = x.dtype)
	
	for k, grad_k in enumerate(x_grad):

		tmp_grad = np.gradient(grad_k)

		for l, grad_kl in enumerate(tmp_grad):

			hessian[k, l, :, :] = grad_kl

	return hessian


# ========================================================= NEURAL NETWORK FUNCTIONS ================================================================= #


# ==== Diferent sigmoid functions (with respective gradients): =============

def sigmoid_logistic_function(z): # Activation function: logistic function

	# These two lines in loops are useful to prevent overflows...
	if z < -45:
		return 0.
	elif z > 45:
		return 1.

	else:
		return 1. / (1. + np.exp(-1.*z) )
sigmoid_logistic_function = np.vectorize(sigmoid_logistic_function)

def sigmoid_logistic_function_grad(z): # Gradient of the sigmoid function: logistic function

	return sigmoid_logistic_function(z) * (1. - sigmoid_logistic_function(z) )


def sigmoid_hyperbolic_tangent(z): # Activation function: hyperbolic tangent ( see Y. LeCun, 1988 )

	a = 1.7159
	b = 2./3.

	# These two lines in loops are useful to prevent overflows...
	if z < -45:
		return -a
	elif z > 45:
		return a

	else:
		return a * np.tanh(b * z)
sigmoid_hyperbolic_tangent = np.vectorize(sigmoid_hyperbolic_tangent)


def sigmoid_hyperbolic_tangent_grad(z): # Gradient of the sigmoid function: hyperbolic tangent ( see Y. LeCun, 1988 )

	a = 1.7159
	b = 2./3.

	return b/a * (a - sigmoid_hyperbolic_tangent(z)) * (a + sigmoid_hyperbolic_tangent(z))
sigmoid_hyperbolic_tangent_grad = np.vectorize(sigmoid_hyperbolic_tangent_grad)


def map_hyperbolic_tangent(h): # maps the output of the hyperbolic tangent sigmoid function into the [0,1] range for a probabilistic interpretation of its output
	return 1./(1.7159*2.)*(h + 1.7159)


def random_initialization_parameter(Lin,Lout,sigmoid): # Limit on the random number for random initialization of parameters for a neural network

	if sigmoid == sigmoid_hyperbolic_tangent:
		return Lin**(-1./2.)

	else:
		return np.sqrt(6./ (Lin+Lout))



def reshape_matrix(theta, M, n, m1, m2): # Reshapes a 1-D parameter vector in the M+1 parameter matrices for a M+2-layers neural network

	theta_1 = theta[ 0 : (m1*(n+1)) ]
	theta_1 = theta_1.reshape([m1,n+1])

	if M==1: # one hidden layer

		theta_2 = theta[ (m1*(n+1)) : ]
		theta_2 = theta_2.reshape([1,m1+1])

		return theta_1, theta_2

	if M==2: # two hidden layers

		theta_2 = theta[ (m1*(n+1)) : (m1*(n+1)) + (m2*(m1+1)) ]
		theta_2 = theta_2.reshape([m2,m1+1])

		theta_3 = theta[ (m1*(n+1)) + (m2*(m1+1))  : ]
		theta_3 = theta_3.reshape([1,m2+1])

		return theta_1, theta_2, theta_3


def forward_prop_hyp(theta, x, M, n, m1, m2, sigmoid): #  Forward propagation to get the hypothesis for a SINGLE TRAINING EXAMPLE of a neural-network

	if M==1: # just one hidden layer
		Theta_1, Theta_2 = reshape_matrix(theta,M,n,m1,m2)

	if M==2: # two hidden layers
		Theta_1, Theta_2, Theta_3 = reshape_matrix(theta,M,n,m1,m2)

	# Hidden layer:
	z_2 = np.dot(Theta_1, x)
	a_2 = sigmoid(z_2)

	a_2 = np.insert(a_2,0,1.) # Adding the bias unit

	# Output layer (or second layer if M==2):
	z_3 = np.dot(Theta_2, a_2)
	a_3 = sigmoid(z_3) # nn hypothesis

	if M == 1:

		return z_2,a_2,z_3,a_3

	if M == 2:

		# second hidden layer:
		a_3 = np.insert(a_3,0,1.) # Adding the bias unit
		
		# Output layer:
		z_4 = np.dot(Theta_3, a_3)
		a_4 = sigmoid(z_4)
		
		return z_2,a_2,z_3,a_3,z_4,a_4



def forward_prop_cost_single(theta, x, y, M, n, m1, m2, sigmoid): # Forward propagation to get the NON-regularized cost function for a SINGLE training example in a neural-network

	if M == 1: # just one hidden layer

		_,_,h = forward_prop_hyp(theta, x, M, n, m1, m2, sigmoid)

	if M == 2: # two hidden layers

		_,_,_,_,_,h = forward_prop_hyp(theta, x, M, n, m1, m2, sigmoid) # nn hypothesis

	
	return 1./2. * ( y - h )**2. # Unregularized cost function for a single example ( total instantaneous error energy )



def forward_prop_cost_single_reg(theta, x, y, M, n, m, m1, m2, lam_reg, sigmoid): # Regularized cost function for a SINGLE training example in a neural-network

	J_non_reg = forward_prop_cost_single(theta, x, y, M, n, m1, m2, sigmoid) # NON-regularized cost function for a SINGLE training example in a neural-network

	if M == 1: # just one hidden layer
	
		theta_1, theta_2 = reshape_matrix(theta,M,n,m1,m2)

		return J_non_reg + lam_reg/(2.*m) * ( np.sum(theta_1[:,1:]**2.) + np.sum(theta_2[:,1:]**2.) )

	if M == 2: # two hidden layers

		theta_1, theta_2, theta_3 = reshape_matrix(theta,M,n,m1,m2)

		return J_non_reg + lam_reg/(2.*m) * ( np.sum(theta_1[:,1:]**2.) + np.sum(theta_2[:,1:]**2.) + np.sum(theta_3[:,1:]**2.) ) 



def forward_prop_cost(theta, X, Y, M, n, m1, m2, lam_reg, sigmoid): # Forward propagation to get the TOTAL regularized cost function for a neural-network

	m = Y.size

	J = [ forward_prop_cost_single(theta, xi, yi, M, n, m1, m2, sigmoid) for (xi,yi) in zip(X,Y)] # UN-regularized cost function

	if M == 1: # just one hidden layer
	
		theta_1, theta_2 = reshape_matrix(theta,M,n,m1,m2)

		return np.mean(J) + lam_reg/(2.*m) * ( np.sum(theta_1[:,1:]**2.) + np.sum(theta_2[:,1:]**2.) )

	if M == 2: # two hidden layers

		theta_1, theta_2, theta_3 = reshape_matrix(theta,M,n,m1,m2)

		return np.mean(J) + lam_reg/(2.*m) * ( np.sum(theta_1[:,1:]**2.) + np.sum(theta_2[:,1:]**2.) + np.sum(theta_3[:,1:]**2.) ) 



def forward_prop_PSO(params, Xt, Yt, Xval, Yval, M, n, lam_reg, eps, sigmoid):

	m1 = params[0] # Number of neurons in the first hidden layer
	m2 = params[1] # Number of neurons in the second hidden layer
	alpha = params[2] # Global learning rate
	#lam_reg = params[3] ??

	m1 = np.rint(m1)
	m2 = np.rint(m2)

	m = Yt.size # number of training examples
	m_val = Yval.size # number of CV examples

	eps_1 = random_initialization_parameter(n+1, m1, sigmoid)
	Theta_1 = np.random.rand(m1, n+1) * 2.*eps_1 - eps_1 # m1 * (n+1) random matrix

	eps_2 = random_initialization_parameter(m1+1, m2, sigmoid)
	Theta_2 = np.random.rand(m2, m1+1) * 2.*eps_2 - eps_2 # m2 * (m1+1) random matrix

	eps_3 = random_initialization_parameter(m2+1, 1, sigmoid)
	Theta_3 = np.random.rand(1, m2+1) * 2.*eps_3 - eps_3 # 1 * (m2+1) random matrix

	theta = np.concatenate( (Theta_1.ravel(), Theta_2.ravel(), Theta_3.ravel()) ) # Unrolling the three parameter matrices into a single parameter vector	

	J = []
	grad = 0

	N_iter = 5

	for j in range(0,N_iter):

		Xt, Yt = shuffle_in_unison_2(Xt, Yt) # Random shuffling the training examples before each stochastic gradient run

		for i in range(0,m): # loop on training examples

			J_grad = backprop_single_grad_reg(theta, Xt[i], Yt[i], M, n, m, m1, m2, lam_reg, sigmoid) # Gradient of the cost function for a single training example

			grad += J_grad**2.

			theta -= alpha / np.sqrt(grad) * J_grad # AdaGrad optimization, (Duchi, Hazan and Singer 2011)

	#J = forward_prop_cost(theta, Xval, Yval, M, n, m1, m2, lam_reg, sigmoid) # Evaluating the cost function on the cross-validation set (Optimization objective)

	hyp = np.empty(m_val)

	for i in range(m_val):
		_,_,_,_,_,hyp[i] = forward_prop_hyp(theta, Xval[i], M, n, m1, m2, sigmoid) # Evaluating the hypothesis on the cross-validation set

	TP, FP, FN, TN = confusion_matrix(hyp, Yval, eps)  # Confusion matrix (or contingency table)
	M_C_C = MCC(TP, FP, FN, TN) # Matthews Correlation Coefficient [-1, 1]

	print 'm1:,', m1, 'm2:', m2, 'MCC:', M_C_C

	return 1. - M_C_C # Value to minimize (Optimization objective)


def backprop_single_grad(theta, x, y, M, n, m1, m2, sigmoid): #  Backward propagation to get the gradient of the NON-regularized cost function for a SINGLE EXAMPLE in a neural network
	
	# Setting the correct gradients of the sigmoid function
	if sigmoid == sigmoid_hyperbolic_tangent:
		sigmoid_grad = sigmoid_hyperbolic_tangent_grad
	if sigmoid == sigmoid_logistic_function:
		sigmoid_grad = sigmoid_logistic_function_grad


	if M == 1: # just one hidden layer

		Theta_1, Theta_2 = reshape_matrix(theta,M,n,m1,m2)

		a_1 = x # Input layer's value equal to the t-th training example
		
		z_2,a_2,z_3,a_3 = forward_prop_hyp(theta, a_1, M, n, m1, m2, sigmoid) # Feedforward pass for computing the activations for layers 2 and 3

		#Output layer:		
		delta_3 = (a_3 - y) * sigmoid_grad(z_4)
	
		#Hidden layer:
		delta_2 = Theta_2[:,1:] * delta_3 * sigmoid_grad(z_2)

		delta_2 = delta_2.reshape(delta_2.size,1)
		delta_3 = delta_3.reshape(delta_3.size,1)

		a_1 = a_1.reshape(a_1.size,1)
		a_2 = a_2.reshape(a_2.size,1)

		return np.dot(delta_2, a_1.T), np.dot(delta_3, a_2.T)

		#return a_1, a_2, delta_2, delta_3

	if M == 2: # two hidden layers

		Theta_1, Theta_2, Theta_3 = reshape_matrix(theta,M,n,m1,m2)

		a_1 = x # Input layer's value equal to the t-th training example

		z_2,a_2,z_3,a_3,z_4,a_4 = forward_prop_hyp(theta, a_1, M, n, m1, m2, sigmoid) # Feedforward pass for computing the activations for layers 2,3 and 4
		

		#Output layer:
		delta_4 = (a_4 - y) * sigmoid_grad(z_4)

		# 2nd hidden layer:
		delta_3 = np.dot(delta_4, Theta_3[:,1:]) * sigmoid_grad(z_3)

		# 1st hidden layer
		delta_2 = np.dot(delta_3, Theta_2[:,1:]) * sigmoid_grad(z_2)

		delta_2 = delta_2.reshape(delta_2.size,1)
		delta_3 = delta_3.reshape(delta_3.size,1)
		delta_4 = delta_4.reshape(delta_4.size,1)

		a_1 = a_1.reshape(a_1.size,1)
		a_2 = a_2.reshape(a_2.size,1)
		a_3 = a_3.reshape(a_3.size,1)

		return np.dot(delta_2, a_1.T), np.dot(delta_3, a_2.T), np.dot(delta_4, a_3.T)




def backprop_single_grad_reg(theta, x, y, M, n, m, m1, m2, lam_reg, sigmoid): #  Backward propagation to get the gradient of the Regularized cost function for a SINGLE EXAMPLE in a neural network


	if M == 1: # just one hidden layer
	
		theta_1, theta_2 = reshape_matrix(theta,M,n,m1,m2)

		J_grad_1, J_grad_2 = backprop_single_grad(theta, x, y, M, n, m1, m2, sigmoid) # NON-regularized cost function for a SINGLE training example in a neural-network

		J_grad_1[:,1:] = J_grad_1[:,1:] + lam_reg/m * theta_1[:,1:]
		J_grad_2[:,1:] = J_grad_2[:,1:] + lam_reg/m * theta_2[:,1:]

		return np.concatenate( (J_grad_1.ravel(), J_grad_2.ravel()) )

	if M == 2: # two hidden layers

		theta_1, theta_2, theta_3 = reshape_matrix(theta,M,n,m1,m2)

		J_grad_1, J_grad_2, J_grad_3 = backprop_single_grad(theta, x, y, M, n, m1, m2, sigmoid) # NON-regularized cost function for a SINGLE training example in a neural-network

		J_grad_1[:,1:] = J_grad_1[:,1:] + lam_reg/m * theta_1[:,1:]
		J_grad_2[:,1:] = J_grad_2[:,1:] + lam_reg/m * theta_2[:,1:]
		J_grad_3[:,1:] = J_grad_3[:,1:] + lam_reg/m * theta_3[:,1:]

		return np.concatenate( (J_grad_1.ravel(), J_grad_2.ravel(), J_grad_3.ravel()) )
	


def backprop_grad(theta, X, Y, M, n, m1, m2, lam_reg, sigmoid): # Backward propagation to get the gradient of the regularized cost function for a neural network


	if M == 1: # just one hidden layer

		Theta_1, Theta_2 = reshape_matrix(theta,M,n,m1,m2)

		Delta_1 = np.empty(Theta_1.shape)
		Delta_2 = np.empty(Theta_2.shape)

	if M == 2: # two hidden layers

		Theta_1, Theta_2, Theta_3 = reshape_matrix(theta,M,n,m1,m2)

		Delta_1 = np.empty(Theta_1.shape)
		Delta_2 = np.empty(Theta_2.shape)
		Delta_3 = np.empty(Theta_3.shape)

	m = Y.size

	if M == 1: # just one hidden layer

		for t in range(0,m):

			# Accumulating gradients...
			Delta_1 = Delta_1 + backprop_single_grad(theta, X[t], Y[t], M, n, m1, m2, sigmoid)[0]
			Delta_2 = Delta_2 + backprop_single_grad(theta, X[t], Y[t], M, n, m1, m2, sigmoid)[1]
			
		# Gradient for neural network:
		J_grad_1 = Delta_1/m
		J_grad_2 = Delta_2/m

		#Regularized gradient:
		J_grad_1[:,1:] = J_grad_1[:,1:] + lam_reg/m * Theta_1[:,1:]
		J_grad_2[:,1:] = J_grad_2[:,1:] + lam_reg/m * Theta_2[:,1:]

		return np.concatenate( (J_grad_1.ravel(), J_grad_2.ravel()) )

	if M == 2: # two hidden layers

		for t in range(0,m):

			# Accumulating gradients...
			Delta_1 = Delta_1 + backprop_single_grad(theta, X[t], Y[t], M, n, m1, m2, sigmoid)[0]
			Delta_2 = Delta_2 + backprop_single_grad(theta, X[t], Y[t], M, n, m1, m2, sigmoid)[1]
			Delta_3 = Delta_3 + backprop_single_grad(theta, X[t], Y[t], M, n, m1, m2, sigmoid)[2]

		# Gradient for neural network:
		J_grad_1 = Delta_1/m
		J_grad_2 = Delta_2/m
		J_grad_3 = Delta_3/m

		#Regularized gradient:
		J_grad_1[:,1:] = J_grad_1[:,1:] + lam_reg/m * Theta_1[:,1:]
		J_grad_2[:,1:] = J_grad_2[:,1:] + lam_reg/m * Theta_2[:,1:]
		J_grad_3[:,1:] = J_grad_3[:,1:] + lam_reg/m * Theta_3[:,1:]

		return np.concatenate( (J_grad_1.ravel(), J_grad_2.ravel(), J_grad_3.ravel()) )


def num_grad(theta, X_train, Y_train, M, n, m1, m2, lam_reg, eps, sigmoid):

	K = np.size(theta)

	numgrad = np.zeros(K)
	perturb = np.zeros(K)

	for p in range(0,K):

		perturb[p] = eps
		theta_plus  = theta+perturb
		theta_minus = theta-perturb

		loss_plus  = forward_prop_cost(theta_plus, X_train, Y_train, M, n, m1, m2, lam_reg, sigmoid)
		loss_minus = forward_prop_cost(theta_minus,X_train, Y_train, M, n, m1, m2, lam_reg, sigmoid)

		numgrad[p] = (loss_plus - loss_minus)/(2.*eps)

		perturb[p] = 0

	return numgrad


def layer_learning_rate(t, grad):
	return t * (1. + np.log10(1. + 1./ np.sqrt( np.sum( grad**2. ) )) )



# =========================================================== LOGISTIC REGRESSION FUNCTIONS  =========================================================================== #


def compute_cost(theta,X,y, lam_reg): # Regularized cost fuction for logistic regression

	h = hypothesis(theta,X) # hypothesis

	m = y.size # Number of training examples

	J_0 = np.nan_to_num( -y*np.log(h) ) 
	J_1 = np.nan_to_num( (1.-y)*np.log(1.-h) )

	J = 1./m * np.sum( J_0 - J_1 ) +  lam_reg/(2.*m) * np.sum(theta[1:]**2.)  # Regularized cost function: mean on the training examples of the log-likelihood vector

	return J



def hypothesis(theta, X): # hypothesis for logistic regression

	return sigmoid_logistic_function(np.dot(X,theta))


def compute_grad(theta,X,y, lam_reg):

	h = hypothesis(theta,X) # hypothesis
	error = h - y # difference between label and prediction

	m = y.size # Number of training examples
	
	grad = np.empty(theta.size) # initializing the gradient vector

	grad[0] = 1./m * np.dot(error, X[:,0]) # gradient ( not regularized for j=0)
	grad[1:] = 1./m * np.dot(error, X[:,1:]) + 1.*lam_reg/m * theta[1:] # regularized gradient (for j >= 1 )
	
	return grad


def compute_error(theta, X, y):

	h = hypothesis(theta, X) # hypothesis
	
	m = y.size

	return 1. / (2. * m) * np.sum( (h-y)**2. )


# ============================================================== PERFORMANCE EVALUATION FUNCTIONS ====================================================================== #


def confusion_matrix(h, Y, eps):

	TP = 1.*np.size( np.where( (h > eps) & (Y == 1) ) ) # number of True Positives
	FP = 1.*np.size( np.where( (h > eps) & (Y == -1) ) ) # number of False Positives
	FN = 1.*np.size( np.where( (h < eps) & (Y == 1) ) ) # number of False Negatives
	TN = 1.*np.size( np.where( (h < eps) & (Y == -1) ) ) # number of True Negatives

	return TP, FP, FN, TN



def F1_score(TP,FP,FN): # F1 score, [0,1]

	P = TP / (TP + FP) # Precision
	R = TP / (TP + FN) # Recall

	F1 = 2.*P*R / (P+R) # F1 score

	return F1


def MCC(TP,FP,FN,TN): # Matthews Correlation Coefficient, [-1,1]

	return ( TP*TN - FP*FN ) / np.sqrt( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) )


def feature_scaling(x): # Scales the feature x to have approximately zero mean and in range [-1,1]

	x_mean = np.mean(x) # Feature mean
	x_std = np.std(x) # feature standard deviation

	return (x - x_mean) / x_std, x_mean, x_std



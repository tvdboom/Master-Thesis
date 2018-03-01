"""
-------------------------------------------------------------------------------
			    Major master research
			  Marco Trueba van den Boom
-------------------------------------------------------------------------------
                                     MAIN
-------------------------------------------------------------------------------
"""

import sys
sys.path.append('./Functions')

#import standard packages
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import time

from joblib import Parallel, delayed
import multiprocessing

#to import codes from functions folder
import sys
sys.path.append('./Codes')

#machine learning packages
import ml_functions as ml
from particle_swarm_optimization import *
import scipy.optimize as opt


t_init = time.time() # To measure the time the whole process takes


def load_data(path, percentage):
	
	# ================ Reading the Dataset ======================= #

	print '\nReading in data...'
	
	# Training set: labeled data on which the NN learns the classification rule
	X_train = np.load(path + "Training_Set.npy")
	Y_train = np.load(path + "Training_Set_output.npy")
	L_train = np.load(path + "Training_Set_labels.npy")
	
	
	# Validation set: labeled data to choose the best values for the hyper-parameters (m1, m2, eta, lam_reg)
	X_val = np.load(path + "CrossVal_Set.npy")
	Y_val = np.load(path + "CrossVal_Set_output.npy")
	L_val = np.load(path + "CrossVal_Set_labels.npy")

	# Test set: labeled data to evaluate the performance of the algorith,
	X_test = np.load(path + "Test_Set.npy")
	Y_test = np.load(path + "Test_Set_output.npy")
	L_test = np.load(path + "Test_Set_labels.npy")

	
	# Shuffling randomly the data sets (essential for stochastic gradient descent)
	X_train, Y_train, L_train = ml.shuffle_in_unison_3(X_train, Y_train, L_train) 
	X_val, Y_val, L_val = ml.shuffle_in_unison_3(X_val, Y_val, L_val)
	X_test, Y_test, L_test = ml.shuffle_in_unison_3(X_test, Y_test, L_test)
	
	return X_train[:len(X_train)*percentage/100], Y_train[:len(Y_train)*percentage/100], L_train[:len(L_train)*percentage/100], X_val[:len(X_val)*percentage/100], \
	 Y_val[:len(Y_val)*percentage/100], L_val[:len(L_val)*percentage/100], X_test[:len(X_test)*percentage/100], Y_test[:len(Y_test)*percentage/100], L_test[:len(L_test)*percentage/100]



def run_particle_swarm_optimization(X_train, Y_train, X_val, Y_val, M, n, lam_reg, eps, activation_function):
	
	# Particle Swarm Optimization (PSO) (VERY SLOW!!)
	# Determine (on the CV set) the best values for the hyper-parameters m1, m2, eta and lam_reg (VERY slow)
	# Finding the best hyper-parameter values which minimize the cross-validation error
	# Problem: it's very slow, almost impossible to apply to the complete training set so it is applied to a subset
	
	t_pso = time.time() # To measure the time the PSO takes
	
	print '\nRunning particle swarm optimization...'
	
	red_number = 1000	   # reduced number of labeled examples for the PSO, otherwise it takes too long to run.... (typical number: ~ 1000)

	Xt = X_train[ : red_number]
	Yt = Y_train[ : red_number]

	Xcv = X_val[ : red_number]
	Ycv = Y_val[ : red_number]

	lb = [10, 10, 0.001]   # lower bounds on the parameters (number of neurons in the 1st layer, number of neuros in the 2nd layer, global learning rate)
	ub = [200, 200, 1]     # upper bounds on the parameters (number of neurons in the 1st layer, number of neuros in the 2nd layer, global learning rate)

	xopt, fopt = pso(ml.forward_prop_PSO, lb, ub, args = (Xt, Yt, Xcv, Ycv, M, n, lam_reg, eps, activation_function), swarmsize = 20, maxiter = 20, debug=True) # pso imported from pyswarm package

	# Optimal parameters:
	m1 =  np.rint(xopt[0]) # number of neurons in the 1st hidden layer (not including the bias term)
	m2 =  np.rint(xopt[1]) # number of neurons in the 2nd hidden layer (not including the bias term)
	eta = xopt[2] 		   # global learning rate
	
	print "\nParticle swarm optimization took %i seconds = %s hours" %(time.time()-t_pso, round((time.time()-t_pso)/60./60. ,3) )
	
	return int(m1), int(m2), eta



def run_gradient_checking(theta, X_train, Y_train, M, n, m1, m2, lam_reg, activation_function):
	
	# Check that backpropagation is implemented correctly (VERY SLOW!!)
	
	t_grad = time.time() # To measure the time gradient checking takes
	
	print '\nRunning gradient checking...'
	
	eps_gc = 1.e-6 # small constant to compute numerical derivatives

	J_grad = ml.backprop_grad(theta, X_train, Y_train, M, n, m1, m2, lam_reg, activation_function) 		# Exact gradient, computed with backprop

	numgrad = ml.num_grad(theta, X_train, Y_train, M, n, m1, m2, lam_reg, eps_gc, activation_function) 	# Approximate gradient
	
	print "\nGradient checking took %i seconds = %s hours" %(time.time()-t_grad, round((time.time()-t_grad)/60./60. ,3) )
	
	np.save('grad_check.npy', zip(J_grad, numgrad))

	return J_grad, numgrad



def performance_evaluation(theta_opt, X_test, Y_test, eps, M, n, m1, m2, activation_function):
	
	# =========== Error analysis and Performance evaluation (on TEST set) ====================== #

	m_test = X_test.shape[0] # Number of examples in the test set
	hyp = np.empty(m_test)

	if M == 1: # just one hidden layer
		for i in range(m_test):
			_,_,_,hyp[i] = ml.forward_prop_hyp(theta_opt, X_test[i], M, n, m1, m2, activation_function) # Forward propagation to get the nn hypothesis on the test set

	if M == 2: # two hidden layers
		for i in range(m_test):
			_,_,_,_,_,hyp[i] = ml.forward_prop_hyp(theta_opt, X_test[i], M, n, m1, m2, activation_function) # Forward propagation to get the nn hypothesis on the test set

	TP, FP, FN, TN = ml.confusion_matrix(hyp, Y_test, eps) # Confusion matrix (or contingency table)

	F1  = ml.F1_score(TP, FP, FN) # F1 Score [0, 1]
	MCC = ml.MCC(TP, FP, FN, TN)  # Matthews Correlation Coefficient [-1, 1]

	return F1, MCC



def write_output(path_out, M, m1, m2, eta, lam_reg, theta, J_train, F1, MCC, run, t_end):
	
	# ================== Writing the parameters to a file ==================== #
	
	if M ==1:
		np.savetxt(path_out + 'nn_parameters_' + str(run) + '.txt', (m1, eta, lam_reg, J_train[-1], F1[-1], MCC[-1], t_end), fmt = '%.5g', delimiter = '\n',
		header = 'Parameters of the neural network:\nNumber of neurons in the hidden layer\nLearning rate\nRegularization parameter\nJmin training set\nF1\nMCC\nTime (hours)\n------------------------------')
	elif M == 2:
		np.savetxt(path_out + 'nn_parameters_' + str(run) + '.txt', (m1, m2, eta, lam_reg, J_train[-1], F1[-1], MCC[-1], t_end), fmt = '%.5g',
		header = 'Parameters of the neural network:\nNumber of neurons in the first hidden layer\nNumber of neurons in the second hidden layer\nLearning rate\nRegularization parameter\nJmin training set\nF1\nMCC\nTime (hours)\n------------------')
	
	results = [theta, J_train, F1, MCC]
	np.save(path_out + 'results' + str(run) + '.npy', results)





def run_neural_network(run, particle_swarm_optimization, gradient_checking, stochastic_gradient, activation_function, learning_rate):
    
    '''
    DESCRIPTION ----------------------------------------------------------------------------------------
    
    Core code. Runs the neural network, i.e. it trains and saves the data after
    
    
    ATTRIBUTES -----------------------------------------------------------------------------------------
    
    run --> number of times to run the code
    
    particle_swarm_optimization (boolean) --> True: Run the particle swarm optimization (PSO) (SLOW!)
					      False: Don't run the PSO (default)
    
    gradient_checking (boolean) --> True: Perform gradient checking (SLOW!)
				    False: Dont't perform gradient checking (default)
    
    stochastic_gradient (boolean) --> True: Perform stochastic gradient descent
				      False: Perform batch gradient descent (batch size equal to train size, so no mini-batch!) (SLOW!)
    
    activation_function (boolean) --> True: Use the sigmoid logistic activation function
				      False: Use the hyperbolic tangent activation function
    
    learning_rate --> 1: Update parameters with the AdaGrad method
		      2: Update parameters with the layer-specific AdaGrad method 
		      3: Update parameters with standard stochastic gradient descent
    
    '''
    
    
    percentages = np.linspace(100, 100, 1).astype(int) # percentages of the data to run
    for percentage in percentages:
		    
	    # ============================ Load data ============================== #
	    
	    path = '/disks/strw14/TvdB/Master2/Dataset/' # directory from which to load the data
	    X_train, Y_train, L_train, X_val, Y_val, L_val, X_test, Y_test, L_test = load_data(path, percentage)
	    
	    print '\nTraining with %.5g percentage of the data' %percentage
	    print '---------------------------------------------------------'
	    print 'Number of stars in the training set:', len(X_train)
	    print 'Number of stars in the cross-validation set:', len(X_val)
	    print 'Number of stars in the test set:', len(X_test)

	    
	    
	    
	    # ====================== Initialize parameters ======================== #
	    
	    print '\nInitializing parameters...'
	    
	    path_out = '/disks/strw14/TvdB/Master2/10Runs/'	# directory to output files
	    
	    
	    # Set hyperparameters
	    lam_reg = 0. 	# Regularization parameter

	    eta = 0.071		# Global learning rate

	    m1 = 119 		# Choose the number of neurons in the 1st hidden layer (not including the bias term)
	    m2 = 95 		# Choose the number of neurons in the 2nd hidden layer (not including the bias term) (set 0 if M = 1 !)
	    
	    
	    M = 2 if m2 > 0 else 1 	# Number of hidden layers
	    L = 2 + M 			# Total number of layers in the network (input layer + M hidden layer(s) + output layer)
	    
	    m,n = X_train.shape		# m: number of training examples, n: number of features + 1
	    m_val = X_val.shape[0]		# number of CV examples
	    m_test = X_test.shape[0]	# number of test examples
	    n = n - 1			# number of features (removing the bias feature)
	    
	    
	    # Initialize activation function
	    if activation_function:
		    
		    activation_function = ml.sigmoid_hyperbolic_tangent
		    eps = 0. # Threshold parameter for accepting a HVS  (typically 0 in the case of an odd activation function (f(v) = -f(-v)), such as the hyperbolic tangent function)
		    
		    # For the hyperbolic sigmoid function the negative values have to be -1 since it is an odd function
		    Y_train[Y_train == 0] = -1
		    Y_val[Y_val == 0] = -1
		    Y_test[Y_test == 0] = -1
		    
	    else:
		    
		    activation_function = ml.sigmoid_logistic_function
		    eps = 0.5 # Threshold parameter for accepting a HVS (typically 0.5 in the case of the logistic funtion)

		    
	    # Optional. Get hyperparameters m1, m2, eta with pyswarm
	    if particle_swarm_optimization: 
		    m1, m2, eta = run_particle_swarm_optimization(X_train, Y_train, X_val, Y_val, M, n, lam_reg, eps, activation_function)
	    
	    
	    
	    print '\n--------------  Parameters used ------------------'
	    print 'Number of neurons in the first hidden layer:', m1
	    print 'Number of neurons in the second hidden layer:', m2
	    print 'Global learning rate:', eta
	    print '--------------------------------------------------'


	    # Initialize random values for the weights of the neural network
	    eps_1 = ml.random_initialization_parameter(n+1, m1, activation_function)
	    Theta_1 = np.random.rand(m1, n+1) * 2* eps_1 - eps_1 # m1 * (n+1) random matrix

	    if M ==1:
		    eps_2 = ml.random_initialization_parameter(m1+1, 1, activation_function)
		    Theta_2 = np.random.rand(1, m1+1) * 2*eps_2 - eps_2 # 1 * (m1+1) random matrix
		    theta = np.concatenate( (Theta_1.ravel(), Theta_2.ravel()) ) # Unrolling the two parameter matrices into a single parameter vector

	    if M == 2:
		    eps_2 = ml.random_initialization_parameter(m1+1, m2, activation_function)
		    Theta_2 = np.random.rand(m2, m1+1) * 2*eps_2 - eps_2 # m2 * (m1+1) random matrix
		    eps_3 = ml.random_initialization_parameter(m2+1, 1, activation_function)
		    Theta_3 = np.random.rand(1, m2+1) * 2*eps_3 - eps_3 # 1 * (m2+1) random matrix
		    theta = np.concatenate( (Theta_1.ravel(), Theta_2.ravel(), Theta_3.ravel()) ) # Unrolling the three parameter matrices into a single parameter vector
	    
	    
	    
	    
	    
		    
	    # ========================= Run neural network training ============================ #
	    
	    print '\nTraining neural network...'
	    
	    if stochastic_gradient: # learn parameters with stochastic (online) supervised algorithm

		    n_iter = 0		# index to keep count of the total number of iterations
		    n_epochs = 8 	# repeating n_epochs times the optimization process over the whole training set
		    
		    grad = 0	 	# initializing to zero the value of the gradient
		    
		    n_J = 20		# computing n_J times the cost function of the training set for every epoch. To plot learning curves.
		    plot_steps = np.linspace(1, m*n_epochs, n_J*n_epochs).astype(int)

		    theta_save, J_train, F1, MCC = [], [], [], [] # parameters that are being saved n_J times every epoch (for learning curves)
		    
		    
		    # Allocate memory for gradient for the layer-specific AdaGrad method
		    if learning_rate == 2:
			    grad_layer = np.empty(np.size(theta))
			    t_layer = np.empty(np.size(theta))
	    

		    for j in range(1, n_epochs+1): # for every iteration, start at 1
			    
			    print '\nComputing epoch:', j, '/', n_epochs

			    X_train, Y_train, L_train = ml.shuffle_in_unison_3(X_train, Y_train, L_train) # Random shuffling the training examples before each stochastic gradient run

			    for i in range(m): # for every training example
				    
				    J_grad = ml.backprop_single_grad_reg(theta, X_train[i], Y_train[i], M, n, m, m1, m2, lam_reg, activation_function) # Cost function gradient for a single training example (with regularization)

				    if learning_rate == 1: # Updating parameters with the AdaGrad method (Duchi,Hazan and Singer 2011) 

					    grad += J_grad**2.
					    theta -= eta / np.sqrt(grad) * J_grad


				    elif learning_rate == 2: # Updating parameters with the layer-specific AdaGrad method (Singh, De, ZHang, Goldstein and Taylor 2015)

					    grad += J_grad**2.

					    grad_layer[ : (n+1)*m1 ] = np.sqrt( np.sum( grad[ : (n+1)*m1 ] ) )
					    grad_layer[ (n+1)*m1 : (n+1)*m1 + (m1+1)*m2 ] = np.sqrt( np.sum( grad[ (n+1)*m1 : (n+1)*m1 + (m1+1)*m2 ] ) )
					    grad_layer[ (n+1)*m1 + (m1+1)*m2 : (n+1)*m1 + (m1+1)*m2 + m2+1] = np.sqrt( np.sum( grad[ (n+1)*m1 + (m1+1)*m2 : (n+1)*m1 + (m1+1)*m2 + m2+1] ) )

					    t_layer[ : (n+1)*m1 ] = ml.layer_learning_rate(eta, J_grad[ : (n+1)*m1 ])
					    t_layer[ (n+1)*m1 : (n+1)*m1 + (m1+1)*m2 ] = ml.layer_learning_rate(eta, J_grad[ (n+1)*m1 : (n+1)*m1 + (m1+1)*m2 ])
					    t_layer[ (n+1)*m1 + (m1+1)*m2 : (n+1)*m1 + (m1+1)*m2 + m2+1] = ml.layer_learning_rate(eta, J_grad[ (n+1)*m1 + (m1+1)*m2 : (n+1)*m1 + (m1+1)*m2 + m2+1])
					    
					    theta -= t_layer/grad_layer * J_grad


				    else: # Updating parameters with standard stochastic gradient descent

					    theta -= eta * J_grad	

				    n_iter += 1 # index to keep count of the number of iterations
				    
				    
				    # Compute the cost function of the different sets to plot the learning curves
				    if n_iter in plot_steps:
					    print 'iteration:', n_iter, ', ',
					    
					    # Value of the cost function on the training set
					    J_train.append(ml.forward_prop_cost(theta, X_train, Y_train, M, n, m1, m2, lam_reg, activation_function))
					    theta_save.append(theta)
					    
					    # Run evaluation of the performance (on the test set!)
					    F1_now, MCC_now = performance_evaluation(theta, X_test, Y_test, eps, M, n, m1, m2, activation_function)
					    F1.append(F1_now)
					    MCC.append(MCC_now)
					    
			    print 'Time needed to compute epoch:', round((time.time()-t_init)/60./60., 3)
		    
		    
		    # ========================= End of the training ============================
		    
		    
		    
		    # End parameters
		    Jmin_train = J_train[-1]
		    F1min = F1[-1]	# always on test set
		    MCCmin = MCC[-1]	# always on test set
		    t_end = round((time.time()-t_init)/60./60., 3) # endtime in hours


	    else: # Batch Learning, with a Limited-memory BFGS (L-BFGS_B) algorithm (VERY VERY slow!!!)

		    theta_bfgs = opt.fmin_l_bfgs_b(ml.forward_prop_cost, theta, fprime = ml.backprop_grad, args=(X_train, Y_train, M, n, m1, m2, lam_reg, activation_function),maxiter=1000, pgtol=1.e-6) # best parameters
		    theta_opt = theta_bfgs[0]
		    Jmin_train =   theta_bfgs[1]
		    Jmin_test =   0 # ?
		    n_iter =   theta_bfgs[2]['nit']

				    
	    # Optional. Run gradient checking
	    if gradient_checking:
		    J_grad, numgrad = run_gradient_checking(theta, X_train, Y_train, M, n, m1, m2, lam_reg, activation_function)
		    np.save(path_out + 'grad_check' + str(run) + '.npy', [J_grad, numgrad])
	    
			    

	    # Output parameters to file
	    write_output(path_out, M, m1, m2, eta, lam_reg, theta_save, J_train, F1, MCC, run, t_end)


	    # Print results
	    print '\n\n================== Initial Parameters ===================== \n'
	    
	    print 'Total number of layers in the neural network: %s' %(int(L))
	    if M == 1:
		    print 'Number of units in the hidden layer: %s' %(int(m1))
	    if M ==2:
		    print 'Number of units in the 1st hidden layer: %s' %(int(m1))
		    print 'Number of units in the 2nd hidden layer: %s' %(int(m2))
	    print '\nGlobal learning rate: %s' %(round(eta,5))
	    print 'Regularization parameter: %s' %(round(lam_reg,3))
	    print '\n------------------------------------------------------------- \n'
	    print "Number of training examples: %i" %(m)
	    print "Number of cross-validation examples: %i" %(m_val)
	    print "Number of test examples: %i" %(m_test)
	    
	    print '\n======================= Output Parameters =================== \n'

	    print "Value of the cost function at the minimum (on training set): %s" %( round( Jmin_train, 8) )
	    print "Number of epochs: %i" %(n_epochs)
	    print "Learning the parameters took %s hours" %(t_end)
		    
	    print '\n ========= Performance Evaluation on the Test Set =========== \n'
	    print 'F1 = %s' %(round(F1min,3))
	    print 'MCC = %s' %(round(MCCmin,3))
	    print '\n ============================================================ \n'
	    
    
    







def main(runs, particle_swarm_optimization, gradient_checking, stochastic_gradient, activation_function, learning_rate):
 
    parallel = True 	# Run the trainings in parallel or not
    
    if parallel:
	num_cores = multiprocessing.cpu_count() - 2 # number of cores to use for parallelization
	print 'Using %i cores for parallelization' %(num_cores)

	Parallel(n_jobs = num_cores)(delayed(run_neural_network) (run, particle_swarm_optimization, gradient_checking, stochastic_gradient, activation_function, learning_rate) for run in range(runs))
    
    else:
	for run in range(runs):
	    run_neural_network(run, particle_swarm_optimization, gradient_checking, stochastic_gradient, activation_function, learning_rate)



def new_option_parser():
	op = OptionParser()
	
	# Number of trainings to perform
	op.add_option("-r", "--runs", default = 1, dest="runs", help="number of trainings to perform (default: 1)", type='int')
	
	# Apply particle swarm optimization
	op.add_option("-p", "--particle_swarm_optimization", default = False, dest="particle_swarm_optimization", help="perform particle swarm optimization (default: False)", type='string')

	# Apply gradient checking
	op.add_option("-g", "--gradient_checking", default = False, dest="gradient_checking", help="perform gradient checking (default: False)", type='string')


	# Set the gradient type
	# True: learn parameters with stochastic (online) supervised algorithm
	# False: learn parameters with Batch gradient descent. Might be VERY slow!!
	op.add_option("-s", "--stochastic_gradient", default = True, dest="stochastic_gradient", help="perform stochastic gradient descent (default: True)", type='string')

		
	# Choose the activation function
	# True: Hyperbolic Tangent Sigmoid Function, ouptputs values in [-1.7159, 1.7159]  (see Yann LeCun, Efficient BackProp, Springer 1988)
	# False: Logistic Function or Sigmoid Function,  ouptputs values in [0, 1]
	op.add_option("-a", "--activation_function", default = True, dest="activation_function", help="use the hyperbolic logistic function (default: True)", type='string')

		
	# Choose the learning rate type
	# 0: Update the parameters with standard stochastic gradient descent. (Not recommended!)
	# 1: Parameter-specific adaptive learning rate, (Duchi, Hazan and Singer 2011)
	# 2: Layer-specific adaptive learning rate, (Singh, De, Zhang, Goldstein and Taylor 2015)
	op.add_option("-l", "--learning_rate", default = 1, dest="learning_rate", help="choose the learning rate type (default: 1)", type='int')

	return op
	
	


if __name__ == "__main__":
	
	options, arguments = new_option_parser().parse_args()
	main(**options.__dict__)


